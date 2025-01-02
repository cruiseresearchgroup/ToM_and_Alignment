import os
import json
from tqdm import tqdm
import fire
from dataclasses import fields

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


def get_dataset(args, tokenizer, qa_per_layer=False):
    if qa_per_layer:
        QA_DATA = {i: None for i in args.layers_to_optimize}
        for i in args.layers_to_optimize:
            with open(f"controls/{args.control}_layer{i}.json", "r") as f:
                QA_DATA[i] = list(json.load(f).values())[0]
        num_qa = min([len(qa) for qa in QA_DATA.values()])
        assert num_qa == max([len(qa) for qa in QA_DATA.values()])
    else:
        with open(f"controls/{args.control}.json", "r") as f:
            QA_DATA = list(json.load(f).values())[0]
        num_qa = len(QA_DATA)
    data = []
    if args.dataset == "alpaca":
        alpaca_data = load_dataset("tatsu-lab/alpaca")["train"]
        for item in alpaca_data:
            if (
                item["input"] == ""
                and len(item["instruction"].split()) + len(item["output"].split()) < 300
            ):
                data.append((item["instruction"], item["output"]))
    elif args.dataset == "dolly":
        raw_data = load_dataset("databricks/databricks-dolly-15k")["train"]
        for item in raw_data:
            if len(item["instruction"].split()) > 100:
                continue
            if item["context"] == "":
                data.append((item["instruction"], item["response"]))
            elif len(item["context"].split()) < 200:
                data.append(
                    (item["instruction"] + "\n\n" + item["context"], item["response"])
                )
    else:
        raise ValueError("Invalid dataset")

    formatted_data = []

    for item in data:
        for idx in range(num_qa):
            read_prompt_0 = tokenizer.apply_chat_template(
                [{"role": "user", "content": item[0]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            read_prompt_1 = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": item[0]},
                    {"role": "assistant", "content": item[1]},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            for read_prompt in [read_prompt_0, read_prompt_1]:
                formatted_data.append({"read_prompt": read_prompt, "dialog_idx": idx})
    np.random.shuffle(formatted_data)
    formatted_data = formatted_data[: args.samples]
    if qa_per_layer:
        for i, item in enumerate(formatted_data):
            new_item = {j: None for j in args.layers_to_optimize}
            for j in args.layers_to_optimize:
                q, a = QA_DATA[j][item["dialog_idx"]]
                new_item[j] = {
                    "read_prompt": item["read_prompt"],
                    "dialog": BASE_DIALOG
                    + [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ],
                }
            formatted_data[i] = new_item
    else:
        for i, item in enumerate(formatted_data):
            q, a = QA_DATA[item["dialog_idx"]]
            formatted_data[i] = {
                0: {
                    "read_prompt": item["read_prompt"],
                    "dialog": BASE_DIALOG
                    + [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ],
                }
            }
    return formatted_data


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
    if args.eval_prompts != "":
        with open(f"prompts/{args.eval_prompts}.json", "r") as f:
            prompts = json.load(f)
        chats = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        tokenized = tokenizer(chats, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **tokenized,
            max_new_tokens=500,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        for i in range(len(out)):
            prompt, completion = clean_text(tokenizer.decode(out[i]))
            print(f"[PROMPT]: {prompt}")
            print(f"[COMPLETION]: {completion}")
            print("#" * 80)
            completions.append(completion)
        FOLDER = (
            f"out/completions/{args.control}_{args.dataset}_samples{args.samples}"
        )
        if not os.path.exists(FOLDER):
            os.makedirs(FOLDER)
        with open(f"{FOLDER}/{args.eval_prompts}.json", "w") as f:
            json.dump(completions, f, indent=2)
    return completions


def steer(args, decoder_model, tokenizer, **kwargs):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    assert args.qa_per_layer is False

    target_model = get_target_model(args, device=kwargs["device"])
    module_read, module_write = get_modules(target_model, decoder_model, **vars(args))
    optimizer = torch.optim.Adam(target_model.parameters(), lr=args.lr)
    dataset = get_dataset(args, tokenizer)

    losses = []
    for i in tqdm(range(len(dataset) // args.batch_size)):
        batch = dataset[i * args.batch_size : (i + 1) * args.batch_size]
        tokenized_batch = tokenize(
            [item[0] for item in batch],
            tokenizer,
            name=args.target_model_name,
            generate=False,
            mask_all_but_last=True,
            modify_chat_template=args.modify_chat_template,
        )
        idx = np.random.choice(len(module_read))
        out = latent_qa(
            tokenized_batch,
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
    return get_results(args, target_model.eval(), tokenizer)


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
    dataset = get_dataset(args, tokenizer, qa_per_layer=args.qa_per_layer)

    losses = []
    for i in tqdm(range(len(dataset) // args.batch_size)):
        batch = dataset[i * args.batch_size : (i + 1) * args.batch_size]
        tokenized_batch = {}
        for l_idx in dataset[0].keys():
            tokenized_batch[l_idx] = tokenize(
                [item[l_idx] for item in batch],
                tokenizer,
                name=args.target_model_name,
                generate=False,
                mask_all_but_last=True,
                modify_chat_template=args.modify_chat_template,
            )
        tokenized_read = tokenized_batch[l_idx]["tokenized_read"].to(
            target_model.device
        )
        inputs_embeds = target_model.model.model.embed_tokens(tokenized_read.input_ids)
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
                    tokenized_batch[b_idx]["tokenized_write"].to(decoder_model.device),
                    tokenized_batch[b_idx]["read_lengths"],
                    tokenized_batch[b_idx]["write_lengths"],
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


def main(**kwargs):
    args = steer_config()
    update_config(args, **kwargs)
    tokenizer = get_tokenizer(args.target_model_name)
    decoder_model = get_model(
        model_name=args.target_model_name,
        load_peft_checkpoint=args.decoder_model_name,
        device="cuda:1",
    )
    if args.per_layer_loss:
        per_layer_loss(args, decoder_model, tokenizer, device="cuda:0", **kwargs)
    else:
        steer(args, decoder_model, tokenizer, device="cuda:0", **kwargs)


if __name__ == "__main__":
    fire.Fire(main)
