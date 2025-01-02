import os
from dataclasses import fields
import random
import fire
from tqdm import tqdm

import numpy as np
import wandb
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
import sys

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True"  # prevent memory fragmentation
)

from lit.configs.train_config import train_config
from lit.configs.peft_config import lora_config
from lit.utils.dataset_utils import get_dataloaders
from lit.utils.infra_utils import (
    get_logger,
    setup_wandb,
    save_model,
    get_ema,
    update_ema,
    update_config,
    get_tokenizer,
    get_model,
    get_modules,
)
from lit.utils.activation_utils import latent_qa


def main(**kwargs):
    # Get args and setup DDP
    dist.init_process_group("nccl")
    assert torch.cuda.is_available()
    args = train_config()
    update_config(args, **kwargs)
    fsdp_args = None
    if args.use_fsdp:
        from lit.configs.fsdp_config import fsdp_config

        fsdp_args = fsdp_config()
        update_config(fsdp_args, **kwargs)

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    seed = args.seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    logger = get_logger(args, rank)
    wandb_run = None
    if args.use_wandb and rank == 0:
        wandb_run = setup_wandb(args, fsdp_args, **kwargs)
    # Load tokenizer and datasets
    tokenizer = get_tokenizer(args.target_model_name)
    train_dataloader, eval_dataloader = get_dataloaders(args, tokenizer)

    # Load the models
    target_model = get_model(
        args.target_model_name, fsdp_args=fsdp_args, device=device, rank=rank
    )
    lora_params = {
        k.name: getattr(lora_config(), k.name) for k in fields(lora_config())
    }
    peft_config = LoraConfig(**lora_params)
    decoder_model = get_model(
        args.target_model_name,
        peft_config=peft_config,
        fsdp_args=fsdp_args,
        device=device,
        rank=rank,
        distributed_training=True,
    )
    torch.cuda.empty_cache()
    if rank == 0:
        decoder_model.module.print_trainable_parameters()
        if wandb_run is not None and args.load_model_checkpoint == "":
            wandb_run.config.update(peft_config)
    module_read, module_write = get_modules(
        target_model, decoder_model, **args.__dict__
    )
    ema = get_ema(decoder_model.module, decay=args.ema_decay, device=device)

    # Initialize the optimizer and learning rate scheduler
    optimizer = optim.AdamW(
        decoder_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    training_steps = len(train_dataloader) * args.num_epochs
    logger.info(f"Training steps: {training_steps}")
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=training_steps,
    )

    # Start the training
    train_steps = 0

    for epoch in range(args.num_epochs):
        decoder_model.train()
        total_length = len(train_dataloader) // args.gradient_accumulation_steps
        pbar = tqdm(
            colour="blue",
            desc=f"Training Epoch: {epoch+1}",
            total=total_length,
            dynamic_ncols=True,
        )
        for step, batch in enumerate(train_dataloader):
            for key in batch.keys():
                batch[key] = batch[key].to(rank)
            layer_list = np.random.choice(
                len(module_read), args.num_layers_to_sample, replace=False
            )
            for idx in layer_list:
                train_steps += 1
                outputs = latent_qa(
                    batch,
                    target_model,
                    decoder_model,
                    module_read[idx],
                    module_write[idx],
                    tokenizer,
                    truncate_verbs=(args.train_with_verb_mask is not None),
                    shift_position_ids=args.shift_position_ids,
                )
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                if train_steps % args.gradient_accumulation_steps == 0:
                    if (
                        args.gradient_clipping
                        and args.gradient_clipping_threshold > 0.0
                    ):
                        torch.nn.utils.clip_grad_norm_(
                            decoder_model.parameters(),
                            args.gradient_clipping_threshold,
                        )
                    optimizer.step()
                    optimizer.zero_grad()
                    update_ema(ema, decoder_model.module, decay=args.ema_decay)
                    pbar.update(1)

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/epoch": epoch,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss.detach().float(),
                    }
                )

            pbar.set_description(
                f"Training Epoch: {epoch+1}/{args.num_epochs}, batch {step+1}/{len(train_dataloader)} completed (loss: {loss.detach().float()})"
            )
            if args.eval_ppl and train_steps % args.eval_every_n_steps == 0:
                assert eval_dataloader is not None
                total_loss = 0.0
                pbar = tqdm(
                    colour="green",
                    desc=f"Evaluating Epoch: {epoch+1}",
                    total=len(eval_dataloader),
                    dynamic_ncols=True,
                )
                for step, batch in enumerate(eval_dataloader):
                    for key in batch.keys():
                        batch[key] = batch[key].to(rank)
                    layer_idx = np.random.choice(len(module_read))
                    outputs = latent_qa(
                        batch,
                        target_model,
                        decoder_model,
                        module_read[layer_idx],
                        module_write[layer_idx],
                        tokenizer,
                        truncate_verbs=(args.train_with_verb_mask is not None),
                        shift_position_ids=args.shift_position_ids,
                        no_grad=True,
                    )
                    total_loss += outputs.loss.detach().float()
                    pbar.update(1)
                losses = torch.zeros(8).to(f"cuda:{rank}")
                losses[rank] = total_loss
                gathered_loss = (
                    [torch.empty_like(losses) for _ in range(dist.get_world_size())]
                    if rank == 0
                    else None
                )
                dist.gather(losses, gathered_loss, dst=0)
                if rank == 0 and wandb_run is not None:
                    all_loss = torch.sum(torch.stack(gathered_loss))
                    all_loss = all_loss / len(eval_dataloader) / dist.get_world_size()
                    wandb_run.log(
                        {
                            "train/epoch": epoch,
                            "train/step": epoch * len(train_dataloader) + step,
                            "eval/loss": all_loss.detach().float(),
                        }
                    )

        # End of epoch
        scheduler.step()
        pbar.close()

        if args.save_model and args.save_after_epoch:
            save_model(
                decoder_model if args.use_fsdp else decoder_model.module,
                ema,
                tokenizer,
                args,
                epoch,
                train_steps,
                logger,
                rank,
            )
            dist.barrier()

    if wandb_run is not None:
        wandb.finish()
    dist.destroy_process_group()
    logger.info("Training completed!")


if __name__ == "__main__":
    fire.Fire(main)
