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

from lit.utils.ToM_reading_utils import *
import sys


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
    
    out = []
    for batch_indices in batch_index_generator(len(questions), args.batch_size):
        print(batch_indices)
        probe_data = []
        for idx in batch_indices:
            rp, qa = dialogs[idx], questions[idx]
            question = [{"role": "user", "content": qa[0]}]
            read_prompt = tokenizer.apply_chat_template(
                        rp,
                        tokenize=False,
                        add_generation_prompt=True,
                        )
            probe_data.append(
            {
                "read_prompt": read_prompt,
                "dialog": BASE_DIALOG + question,
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
        temp = latent_qa(
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
        out+=temp

    responses_data = []
    if generate:
        for i in range(len(out)):
            
            prompt, completion = clean_text(tokenizer.decode(out[i]))
            print(f"[PROMPT]: {questions[i % len(questions)][0]['content']}")
            print(f"[COMPLETION]: {completion}")
            print(f"[GT]: {questions[i][-1]['content']}")
            print("#" * 80)
            
            responses_data.append({
                "index": i,
                "response": completion.strip(),
                "input_prompt": eval(prompt)['content'].strip(),
                "ground_truth": questions[i][-1]['content'].strip()
            })
        if args.save_name != "":
            with open(f"controls/{args.save_name}.jsonl", "w") as f:
                json.dump(responses_data, f, indent=2)
    return responses_data, out, batch


def main(**kwargs):
    from lit.configs.interpret_config import interpret_config
    args = interpret_config()
    update_config(args, **kwargs)
    tokenizer = get_tokenizer(args.target_model_name)
    set_ids = None
    if args.eval_qa.find('CaSiNo')!=-1:
        read_prompts, QAs = get_CaSiNo_text(args, tokenizer, False)
    elif args.eval_qa.find('BARGAIN')!=-1:
        read_prompts, QAs = get_bargain_text(args, tokenizer, False)
    elif args.eval_qa.find('FANTOM')!=-1:
        read_prompts, QAs = get_FanToM_text(args, tokenizer, False)
    elif args.eval_qa.find('NegotiationToM')!=-1:
        read_prompts, QAs = get_NegotiationToM_text(args, tokenizer, False)



    dialogs = read_prompts
    questions = QAs

    
    decoder_model = get_model(
        model_name=args.target_model_name,
        load_peft_checkpoint=args.decoder_model_name,
        device="cuda:0",
    )
    target_model = get_model(args.target_model_name, device="cuda:1")
    
    
    interpret(target_model, decoder_model, tokenizer, dialogs, questions, args)


if __name__ == "__main__":
    fire.Fire(main)
