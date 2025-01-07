import os
import json
from tqdm import tqdm
import fire
from dataclasses import fields
import gc

import numpy as np
import torch
from peft import LoraConfig
from datasets import load_dataset
import matplotlib.pyplot as plt

from lit.utils.dataset_utils import tokenize, BASE_DIALOG
from lit.utils.infra_utils import (
    update_config,
    clean_text,
    get_model,
    get_modules,
    get_tokenizer,
)
from lit.utils.activation_utils import (
    latent_qa,
    generate_substitute_layer_single,
    no_op,
    get_pos_ids,
)
from lit.configs.steer_config import steer_config
from lit.utils.ToM_steering_utils import get_steering_dataloaders, get_evaluation_chats_NegotiationToM, get_evaluation_chats_NegotiationToM_belief


import sys

def get_target_model(args, device):
    lora_params = {
        k.name: getattr(args.peft_config, k.name) for k in fields(args.peft_config)
    }
    args.layers_to_optimize = list(args.layers_to_optimize)
    lora_params["layers_to_transform"] = args.layers_to_optimize
    peft_config = LoraConfig(**lora_params)
    target_model = get_model(
        args.target_model_name, peft_config=peft_config, device=device
    )
    return target_model


def get_results(args, model, tokenizer):
    model.eval()
    completions = []
    if args.save_model:
        FOLDER = f"out/model/steer_{args.control}_{args.dataset}_{args.samples}"
        if not os.path.exists(FOLDER):
            os.makedirs(FOLDER)
        model.save_pretrained(FOLDER)
        args.peft_config = vars(args.peft_config)
        with open(f"{FOLDER}/args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"Model is saved in {FOLDER}")
    # chats, golden_responses = get_evaluation_chats_NegotiationToM('Show-Empathy')
    chats, golden_responses = get_evaluation_chats_NegotiationToM_belief()
    alingned_results = []
    for idx, chat in enumerate(chats):
        print('*'*100)
        # print(idx, chat)
        tokenized = tokenizer.apply_chat_template(chat, return_tensors="pt", padding=True).to(model.device)
        # print(tokenized)
        out = model.generate(
            tokenized,
            max_new_tokens=100,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        temp_response = tokenizer.decode(out[0])
        temp_response = temp_response.split('\n\n')[-1].strip('<|eot_id|>')
        print('&'*100)
        print(temp_response)
        alingned_results.append(temp_response)
        # print(type(out), out.size())
        # print(tokenizer.decode(out[0]))
    return alingned_results, golden_responses
    


def steer(args, decoder_model, tokenizer, **kwargs):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    assert args.qa_per_layer is False
    
    train_dataloader = get_steering_dataloaders(args, tokenizer)
    target_model = get_target_model(args, device=kwargs["device"])
    module_read, module_write = get_modules(target_model, decoder_model, **vars(args))
    optimizer = torch.optim.Adam(target_model.parameters(), lr=args.lr)
    losses = []
    for step, batch in tqdm(enumerate(train_dataloader), colour='blue', desc='Steering Process', total=len(train_dataloader)):
        for key in batch.keys():
            if key.find('read')!=-1:
                batch[key] = batch[key].to("cuda:0")
            if key.find('write')!=-1:
                batch[key] = batch[key].to("cuda:0")
        idx = np.random.choice(len(module_read))
        out = latent_qa(
            batch,
            target_model,
            decoder_model,
            module_read[idx],
            module_write[idx],
            tokenizer,
            shift_position_ids=True,
            generate=False,
            cache_target_model_grad=True,
        )
        loss = out["loss"]
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    plt.plot(losses)
    plt.savefig(f"losses.png")
    aligned_responses, _ = get_results(args, target_model.eval(), tokenizer)
    del target_model
    del decoder_model
    torch.cuda.empty_cache() and gc.collect() 
    target_model = get_target_model(args, device=kwargs["device"])
    nonaligned_responses, golden_responses = get_results(args, target_model.eval(), tokenizer)
    final_data = [{'aligned_respons':ar, 'nonaligned_resopnse':nar, 'golden_response':gr} for ar, nar, gr in zip(aligned_responses, nonaligned_responses, golden_responses)]
    with open('./out/Belief_High_Water.jsonl', 'w') as fout:
        json.dump(final_data, fout)



def per_layer_loss(args, decoder_model, tokenizer, **kwargs):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    
    target_model = get_target_model(args, device=kwargs["device"])
    module_write = [eval("decoder_model.model.model.layers[0]")]
    l_to_optimizer = {
        layer: torch.optim.Adam(
            target_model.model.model.layers[layer].parameters(), lr=args.lr
        )
        for layer in args.layers_to_optimize
    }
    # dataset = get_dataset(args, tokenizer, qa_per_layer=args.qa_per_layer)
    train_dataloader = get_steering_dataloaders(args, tokenizer)
    losses = []
    for step, batch in tqdm(enumerate(train_dataloader), colour='blue', desc='Steering Process', total=len(train_dataloader)):
        for key in batch.keys():
            if key.find('read')!=-1:
                batch[key] = batch[key].to("cuda:0")
            if key.find('write')!=-1:
                batch[key] = batch[key].to("cuda:0")
        inputs_embeds = target_model.model.model.embed_tokens(batch.input_ids)
        cache_position = torch.arange(
            0, inputs_embeds.shape[1], device=inputs_embeds.device
        )   
        position_ids = cache_position.unsqueeze(0)
        causal_mask = target_model.model.model._update_causal_mask(
            tokenized_read.attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values=None,
            output_attentions=False,
        )
        hidden_states = inputs_embeds
        position_embeddings = target_model.model.model.rotary_emb(
            hidden_states, position_ids
        )

        for l_idx, decoder_layer in enumerate(target_model.model.model.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
            if l_idx in l_to_optimizer:
                b_idx = l_idx if args.qa_per_layer else 0
                tokenized_write, read_lengths, write_lengths = (
                    batch["tokenized_write"].to(decoder_model.device),
                    batch["read_lengths"],
                    batch["write_lengths"],
                )
                decoder_position_ids = None
                hidden_states.retain_grad()
                if args.shift_position_ids:
                    decoder_position_ids = get_pos_ids(
                        tokenized_read, tokenized_write
                    ).to(decoder_model.device)
                out = generate_substitute_layer_single(
                    decoder_model,
                    tokenizer,
                    tokenized_write.to(decoder_model.device),
                    module_write,
                    [hidden_states.to(decoder_model.device)],
                    "output",
                    substitute_by_mask=(read_lengths, write_lengths),
                    prepare_inputs=no_op,
                    position_ids=decoder_position_ids,
                    use_cache=False,
                )
                out["loss"].backward()
                if l_idx == 15:
                    losses.append(out["loss"].item())
                # hidden_states -= args.lr * hidden_states.grad
                hidden_states = hidden_states.detach()
                optimizer = l_to_optimizer[l_idx]
                optimizer.step()
                optimizer.zero_grad()
    plt.plot(losses)
    plt.savefig(f"losses.png")
    return get_results(args, target_model.eval(), tokenizer)


    # for i in tqdm(range(len(dataset) // args.batch_size)):
    #     batch = dataset[i * args.batch_size : (i + 1) * args.batch_size]
    #     tokenized_batch = {}
    #     for l_idx in dataset[0].keys():
    #         tokenized_batch[l_idx] = tokenize(
    #             [item[l_idx] for item in batch],
    #             tokenizer,
    #             name=args.target_model_name,
    #             generate=False,
    #             mask_all_but_last=True,
    #             modify_chat_template=args.modify_chat_template,
    #         )
    #     tokenized_read = tokenized_batch[l_idx]["tokenized_read"].to(
    #         target_model.device
    #     )
    #     inputs_embeds = target_model.model.model.embed_tokens(tokenized_read.input_ids)
    #     cache_position = torch.arange(
    #         0, inputs_embeds.shape[1], device=inputs_embeds.device
    #     )
    #     position_ids = cache_position.unsqueeze(0)
    #     causal_mask = target_model.model.model._update_causal_mask(
    #         tokenized_read.attention_mask,
    #         inputs_embeds,
    #         cache_position,
    #         past_key_values=None,
    #         output_attentions=False,
    #     )
    #     hidden_states = inputs_embeds
    #     position_embeddings = target_model.model.model.rotary_emb(
    #         hidden_states, position_ids
    #     )

    #     for l_idx, decoder_layer in enumerate(target_model.model.model.layers):
    #         layer_outputs = decoder_layer(
    #             hidden_states,
    #             attention_mask=causal_mask,
    #             position_ids=position_ids,
    #             past_key_value=None,
    #             cache_position=cache_position,
    #             position_embeddings=position_embeddings,
    #         )
    #         hidden_states = layer_outputs[0]
    #         if l_idx in l_to_optimizer:
    #             b_idx = l_idx if args.qa_per_layer else 0
    #             tokenized_write, read_lengths, write_lengths = (
    #                 tokenized_batch[b_idx]["tokenized_write"].to(decoder_model.device),
    #                 tokenized_batch[b_idx]["read_lengths"],
    #                 tokenized_batch[b_idx]["write_lengths"],
    #             )
    #             decoder_position_ids = None
    #             hidden_states.retain_grad()
    #             if args.shift_position_ids:
    #                 decoder_position_ids = get_pos_ids(
    #                     tokenized_read, tokenized_write
    #                 ).to(decoder_model.device)
    #             out = generate_substitute_layer_single(
    #                 decoder_model,
    #                 tokenizer,
    #                 tokenized_write.to(decoder_model.device),
    #                 module_write,
    #                 [hidden_states.to(decoder_model.device)],
    #                 "output",
    #                 substitute_by_mask=(read_lengths, write_lengths),
    #                 prepare_inputs=no_op,
    #                 position_ids=decoder_position_ids,
    #                 use_cache=False,
    #             )
    #             out["loss"].backward()
    #             if l_idx == 15:
    #                 losses.append(out["loss"].item())
    #             # hidden_states -= args.lr * hidden_states.grad
    #             hidden_states = hidden_states.detach()
    #             optimizer = l_to_optimizer[l_idx]
    #             optimizer.step()
    #             optimizer.zero_grad()

    # plt.plot(losses)
    # plt.savefig(f"losses.png")
    # return get_results(args, target_model.eval(), tokenizer)


def main(**kwargs):
    args = steer_config()
    update_config(args, **kwargs)
    tokenizer = get_tokenizer(args.target_model_name)
    decoder_model = get_model(
        model_name=args.target_model_name,
        load_peft_checkpoint=args.decoder_model_name,
        device="cuda:0",
    )
    if args.per_layer_loss:
        per_layer_loss(args, decoder_model, tokenizer, device="cuda:0", **kwargs)
    else:
        steer(args, decoder_model, tokenizer, device="cuda:0", **kwargs)


if __name__ == "__main__":
    fire.Fire(main)
