import json
import fire

import numpy as np
import torch

from lit.utils.dataset_utils import tokenize, BASE_DIALOG
from lit.utils.activation_utils import latent_qa
from lit.utils.infra_utils import (
    update_config,
    get_model,
    get_tokenizer,
    get_modules,
    clean_text,
)

QUESTIONS = [
    ["What type of persona will the assistant adopt?"],
]


def interpret(
    target_model,
    decoder_model,
    tokenizer,
    dialogs,
    questions,
    args,
    generate=True,
):
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    module_read, module_write = get_modules(target_model, decoder_model, **vars(args))

    if all([len(d) == 1 for d in dialogs]):
        assert args.truncate == "none"
    elif min([len(d) for d in dialogs]) == max([len(d) for d in dialogs]):
        pass
    else:
        assert False

    probe_data = []
    for dialog in dialogs:
        if len(dialog) == 1:
            read_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": dialog[0]}],
                tokenize=False,
                add_generation_prompt=True,
            )
        elif len(dialog) == 2:
            read_prompt = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": dialog[0]},
                    {"role": "assistant", "content": dialog[1]},
                ],
                tokenize=False,
            )
        else:
            read_prompt = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": dialog[0]},
                    {"role": "assistant", "content": dialog[1]},
                    {"role": "user", "content": dialog[2]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        for item in questions:
            if generate:
                dialog = [{"role": "user", "content": item[0]}]
            else:
                dialog = [
                    {"role": "user", "content": item[0]},
                    {"role": "assistant", "content": item[1]},
                ]
            probe_data.append(
                {
                    "read_prompt": read_prompt,
                    "dialog": BASE_DIALOG + dialog,
                }
            )
    batch = tokenize(
        probe_data,
        tokenizer,
        name=args.target_model_name,
        generate=generate,
        get_verb_mask=args.truncate if args.truncate != "none" else None,
        modify_chat_template=args.modify_chat_template,
    )
    out = latent_qa(
        batch,
        target_model,
        decoder_model,
        module_read[0],
        module_write[0],
        tokenizer,
        shift_position_ids=False,
        generate=generate,
        cache_target_model_grad=False,
    )

    QA_PAIRS = {}
    if generate:
        for i in range(len(out)):
            if i % len(questions) == 0:
                curr_dialog = dialogs[i // len(questions)][0]
                QA_PAIRS[curr_dialog] = []
            prompt, completion = clean_text(tokenizer.decode(out[i]))
            print(f"[PROMPT]: {questions[i % len(questions)]}")
            print(f"[COMPLETION]: {completion}")
            print("#" * 80)
            QA_PAIRS[curr_dialog].append((prompt, completion))
        if args.save_name != "":
            with open(f"controls/{args.save_name}.json", "w") as f:
                json.dump(QA_PAIRS, f, indent=2)
    return QA_PAIRS, out, batch


def main(**kwargs):
    from lit.configs.interpret_config import interpret_config

    args = interpret_config()
    update_config(args, **kwargs)
    tokenizer = get_tokenizer(args.target_model_name)
    decoder_model = get_model(
        model_name=args.target_model_name,
        load_peft_checkpoint=args.decoder_model_name,
        device="cuda:0",
    )
    target_model = get_model(args.target_model_name, device="cuda:0")
    dialogs = [[args.prompt]]
    questions = QUESTIONS
    interpret(target_model, decoder_model, tokenizer, dialogs, questions, args)


if __name__ == "__main__":
    fire.Fire(main)
