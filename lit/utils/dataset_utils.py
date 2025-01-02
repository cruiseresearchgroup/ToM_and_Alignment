import random
from itertools import islice
import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.distributed as dist

from .ToM_dataset_utils import *
###################################
###### Tokens and formatting ######
###################################

IGNORE_IDX = -100

NUM_READ_TOKENS_TO_SHIFT = {
    "meta-llama/Meta-Llama-3-8B-Instruct": 1,
    "meta-llama/Meta-Llama-3-70B-Instruct": 1,
    "mistralai/Ministral-8B-Instruct-2410": 2,
    # added unsloth 
    "meta-llama/Llama-3.2-1B-Instruct": 1,
    "meta-llama/Llama-3.2-3B-Instruct": 1, 

}

# Magic numbers that are the length of the user tag + BOS token
NUM_WRITE_TOKENS_TO_SHIFT = {
    "meta-llama/Meta-Llama-3-8B-Instruct": 5,
    "meta-llama/Meta-Llama-3-70B-Instruct": 5,
    "mistralai/Ministral-8B-Instruct-2410": 2,
    # added unsloth
    "meta-llama/Llama-3.2-1B-Instruct": 5,
    "meta-llama/Llama-3.2-3B-Instruct": 5,

}

PAD_TOKEN_IDS = {
    "meta-llama/Meta-Llama-3-8B-Instruct": 128010,
    "meta-llama/Meta-Llama-3-70B-Instruct": 128010,
    "mistralai/Ministral-8B-Instruct-2410": 999,
    # add unsloth
    "meta-llama/Llama-3.2-1B-Instruct": 128010,
    "meta-llama/Llama-3.2-3B-Instruct": 128010, 

    
}

# Magic numbers that correspond to the token idxs of the chat format for the models
CHAT_FORMAT_TOKENS = {
    "meta-llama/Meta-Llama-3-8B-Instruct": (
        torch.tensor([128006, 882, 128007, 271]),
        torch.tensor([128006, 78191, 128007, 271]),
        torch.tensor([128006, 36013, 128007, 271]),
    ),
    "meta-llama/Meta-Llama-3-70B-Instruct": (
        torch.tensor([128006, 882, 128007, 271]),
        torch.tensor([128006, 78191, 128007, 271]),
        torch.tensor([128006, 36013, 128007, 271]),
    ),
    "mistralai/Ministral-8B-Instruct-2410": (
        torch.tensor([3]),
        torch.tensor([4]),
        torch.tensor([4]),
    ),
    # added unsloth
    "meta-llama/Llama-3.2-1B-Instruct": (
        torch.tensor([128006, 882, 128007, 271]),
        torch.tensor([128006, 78191, 128007, 271]),
        torch.tensor([128006, 36013, 128007, 271]),
    ),
    "meta-llama/Llama-3.2-3B-Instruct": (
        torch.tensor([128006, 882, 128007, 271]),
        torch.tensor([128006, 78191, 128007, 271]),
        torch.tensor([128006, 36013, 128007, 271]),
    ),
}

# Mistral should not pass the option --modify_chat_template
DECODER_CHAT_TEMPLATES = {
    "meta-llama/Meta-Llama-3-8B-Instruct": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set role = message['role'] %}{% if role == 'assistant' %}{% set role = 'reflect' %}{% endif %}{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>reflect<|end_header_id|>\n\n' }}{% endif %}",
    "meta-llama/Meta-Llama-3-70B-Instruct": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set role = message['role'] %}{% if role == 'assistant' %}{% set role = 'reflect' %}{% endif %}{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>reflect<|end_header_id|>\n\n' }}{% endif %}",
    # added unsloth 
    "meta-llama/Llama-3.2-1B-Instruct": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set role = message['role'] %}{% if role == 'assistant' %}{% set role = 'reflect' %}{% endif %}{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>reflect<|end_header_id|>\n\n' }}{% endif %}",
    "meta-llama/Llama-3.2-3B-Instruct": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set role = message['role'] %}{% if role == 'assistant' %}{% set role = 'reflect' %}{% endif %}{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>reflect<|end_header_id|>\n\n' }}{% endif %}",
    
}

# Dialog formats for the dataset
BASE_DIALOG = [
    {
        "role": "assistant",
        "content": "Sure, I've analyzed the assistant.",
    }
]

################################
###### Activation masking ######
################################


def mask_inputs(
    input_ids,
    tokenizer_name,
    get_verb_mask=None,
    shift_start=False,
    mask_all_but_last=False,
    modify_chat_template=False,
):
    start_tokens, end_tokens_default, end_tokens_modify = CHAT_FORMAT_TOKENS[
        tokenizer_name
    ]
    end_tokens = end_tokens_modify if modify_chat_template else end_tokens_default
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for b in range(batch_size):
        start_idx = []
        end_idx = []
        for i in range(seq_len):
            if torch.equal(input_ids[b][i : i + len(start_tokens)], start_tokens):
                start_idx.append(i)
            if torch.equal(input_ids[b][i : i + len(end_tokens)], end_tokens):
                end_idx.append(i)

        if get_verb_mask == "user":
            if len(start_idx) == 1:
                continue
            mask[b][start_idx[0] : start_idx[1]] = True
        elif get_verb_mask == "system":
            # start from 1 to exclude <bos> token
            mask[b][1 : start_idx[0]] = True
        # I added it 
        # elif get_verb_mask == "Agent 1":
        #     # start from 1 to exclude <bos> token
        #     mask[b][1 : start_idx[0]] = True
        
        else:
            assert get_verb_mask is None
            if len(start_idx) != len(end_idx):
                # Data is improperly formatted so mask everything and skip this item
                mask[b][:] = True
                continue

            if mask_all_but_last:
                mask[b][: end_idx[-1] + len(end_tokens)] = True
            else:
                for i, (start, end) in enumerate(zip(start_idx, end_idx)):
                    if shift_start and i == 0:
                        mask[b][start - 1 : end + len(end_tokens)] = True
                    else:
                        mask[b][start : end + len(end_tokens)] = True
    return mask


def tokenize(
    batch,
    tokenizer,
    name=None,
    generate=False,
    get_verb_mask=None,
    mask_all_but_last=False,
    modify_chat_template=False,
):
    name = tokenizer.name_or_path if name is None else name
    # added for unsloth
    # tokenizer = get_chat_template(
    #     tokenizer,
    #     chat_template = "llama-3.1",
    # )
    # Tokenize read inputs
    tokenized_read = tokenizer(
        [item["read_prompt"] for item in batch],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    tokenized_batch = {"tokenized_read": tokenized_read}

    # Compute length of read input and maybe add verb_lengths
    read_lengths = torch.sum(tokenized_read.attention_mask, dim=1)
    tokenized_batch["read_lengths"] = read_lengths - 1  # Exclude BOS token

    if get_verb_mask is not None:
        verb_mask = mask_inputs(
            tokenized_read.input_ids, name, get_verb_mask=get_verb_mask
        )
        verb_lengths = torch.sum(verb_mask, dim=1)
        pad_lengths = read_lengths - verb_lengths
        tokenized_batch["verb_lengths"] = verb_lengths
    else:
        pad_lengths = read_lengths

    # Tokenize dialog inputs
    queries = []
    for i in range(len(pad_lengths)):
        query = [
            {
                "role": "user",
                "content": "? " * (pad_lengths[i] - NUM_READ_TOKENS_TO_SHIFT[name]),
            }
        ]
        query += batch[i]["dialog"]
        queries.append(
            tokenizer.apply_chat_template(
                query,
                tokenize=False,
                add_generation_prompt=generate,
                chat_template=(
                    DECODER_CHAT_TEMPLATES[name] if modify_chat_template else None
                ),
            )
        )
    tokenized_write = tokenizer(
        queries,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    tokenized_batch["tokenized_write"] = tokenized_write

    # Compute length of write input
    write_lengths = torch.sum(tokenized_write.attention_mask, dim=1)
    tokenized_batch["write_lengths"] = write_lengths - NUM_WRITE_TOKENS_TO_SHIFT[name]

    # Add labels for training
    if not generate:
        user_inputs_mask = mask_inputs(
            tokenized_write.input_ids,
            name,
            get_verb_mask=None,
            shift_start=any([m in name.lower() for m in ["mistral", "llama-3"]]),
            mask_all_but_last=mask_all_but_last,
            modify_chat_template=modify_chat_template,
        )
        assert tokenizer.padding_side == "left"
        tokenized_write["labels"] = tokenized_write.input_ids.clone()
        mask = (tokenized_write.attention_mask == 0) | user_inputs_mask
        tokenized_write["labels"][mask] = IGNORE_IDX

    return tokenized_batch


###########################
####### Dataloading #######
###########################

# class ToMLatentQADataset(Dataset):
#     def __init__(
#         self,
#         tokenizer,
#         read_prompts,
#         QAs
#     ):
#         self.BD = [
#             {
#                 "role": "assistant",
#                 "content": "Sure, I've analyzed the user.",
#             }
#         ]
#         self.tokenizer = tokenizer
#         self.read_prompts = read_prompts
#         self.QAs = QAs
#         self.lengths = []
#         for rp, qa in zip(read_prompts, QAs):
#             self.lengths.append(sum([len(item['content']) for item in rp])+len(qa[0]['content'])+len(qa[1]['content']))

#     def __len__(self):
#         return len(self.QAs)

#     def __getitem__(self, idx):
#         read_prompt = self.read_prompts[idx]
#         qa_dialog = self.QAs[idx]
#         read_prompt = self.tokenizer.apply_chat_template(
#                 read_prompt, tokenize=False, add_generation_prompt=False
#             )
#         return {"read_prompt": read_prompt, "dialog": self.BD + qa_dialog}

class LatentQADataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_stimulus_completion,
        data_stimulus,
        data_control,
        qa_data,
    ):
        self.tokenizer = tokenizer
        self.data = [data_stimulus_completion[0], data_stimulus[0], data_control[0]]
        self.id_tuples = [
            data_stimulus_completion[1],
            data_stimulus[1],
            data_control[1],
        ]
        self.labels = [
            list(data_stimulus_completion[0].keys()),
            list(data_stimulus[0].keys()),
            list(data_control[0].keys()),
        ]
        self.qa_data = qa_data
        self.lengths = []
        for idx in range(self.__len__()):
            behavior, qa = self.get_behavior_qa(idx)
            self.lengths.append(
                sum([len(s) for s in behavior]) + len(qa[0]) + len(qa[1])
            )
        
    def get_behavior_qa(self, idx):
        if idx < len(self.id_tuples[0]):
            j = 0
        elif idx < len(self.id_tuples[0]) + len(self.id_tuples[1]):
            j = 1
            idx -= len(self.id_tuples[0])
        else:
            j = 2
            idx -= len(self.id_tuples[0]) + len(self.id_tuples[1])
        label_idx, data_idx, qa_idx = self.id_tuples[j][idx]
        label = self.labels[j][label_idx]
        return self.data[j][label][data_idx], self.qa_data[label][qa_idx]

    def __len__(self):
        return sum([len(id_tuples) for id_tuples in self.id_tuples])

    def __getitem__(self, idx):
        behavior, qa = self.get_behavior_qa(idx)
        qa_dialog = [
            {"role": "user", "content": qa[0]},
            {"role": "assistant", "content": qa[1]},
        ]
        control_user, control_model, stimulus_user, stimulus_model = behavior
        if control_model == "":
            assert stimulus_user == stimulus_model == ""
            read_prompt = [{"role": "user", "content": control_user}]
            read_prompt = self.tokenizer.apply_chat_template(
                read_prompt, tokenize=False, add_generation_prompt=True
            )
        elif stimulus_model == "":
            read_prompt = [
                {"role": "user", "content": control_user},
                {"role": "assistant", "content": control_model},
                {"role": "user", "content": stimulus_user},
            ]
            read_prompt = self.tokenizer.apply_chat_template(
                read_prompt, tokenize=False, add_generation_prompt=True
            )
        else:
            read_prompt = [
                {"role": "user", "content": control_user},
                {"role": "assistant", "content": control_model},
                {"role": "user", "content": stimulus_user},
                {"role": "assistant", "content": stimulus_model},
            ]
            read_prompt = self.tokenizer.apply_chat_template(
                read_prompt, tokenize=False, add_generation_prompt=False
            )
        return {"read_prompt": read_prompt, "dialog": BASE_DIALOG + qa_dialog}


class DataCollatorForLatentQA:
    def __init__(
        self,
        tokenizer,
        get_verb_mask,
        generate=False,
        mask_all_but_last=False,
        nudge_persona=False,
        modify_chat_template=False,
    ):
        self.tokenizer = tokenizer
        assert get_verb_mask in ["user", "system", None]
        self.get_verb_mask = get_verb_mask
        self.generate = generate
        self.mask_all_but_last = mask_all_but_last
        self.nudge = "Base your answers on my instructions. " if nudge_persona else ""
        self.modify_chat_template = modify_chat_template

    def __call__(self, batch):
        formatted_batch = []
        for item in batch:
            formatted_batch.append(
                {
                    "read_prompt": item["read_prompt"],
                    "dialog": item["dialog"],
                    "label": item["dialog"][-1]["content"],
                }
            )
        return tokenize(
            formatted_batch,
            self.tokenizer,
            get_verb_mask=self.get_verb_mask,
            generate=self.generate,
            mask_all_but_last=self.mask_all_but_last,
            modify_chat_template=self.modify_chat_template,
        )


class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(
        self, data_source, batch_size: int, drop_last: bool, shuffle: bool = True
    ) -> None:
        self.lengths = data_source.lengths
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        ids = np.argsort(self.lengths)
        if self.drop_last:
            ids = ids[: len(ids) // self.batch_size * self.batch_size]

        batches = [
            ids[i : i + self.batch_size] for i in range(0, len(ids), self.batch_size)
        ]

        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (
                len(self.lengths) % self.batch_size > 0
            )


class DistributedLengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(
        self,
        data_source,
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        random.seed(seed)
        self.batch_sampler = LengthBasedBatchSampler(
            data_source, batch_size=batch_size, drop_last=True, shuffle=shuffle
        )
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        max_length = len(self.batch_sampler) // self.num_replicas * self.num_replicas
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)

    def __len__(self):
        return len(self.batch_sampler) // self.num_replicas


def get_batch_sampler(dataset, train_config, mode):
    return LengthBasedBatchSampler(
        dataset,
        train_config.batch_size_training,
        drop_last=False,
        shuffle=(mode == "train"),
    )


def get_dist_batch_sampler(dataset, train_config, mode):
    return DistributedLengthBasedBatchSampler(
        dataset,
        train_config.batch_size_training,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=(mode == "train"),
        seed=train_config.seed,
    )

# def get_bargain_dataset(train_config, tokenizer, train=True):
#     data_path = train_config.train_qa if train else train_config.eval_qa
#     with open(data_path, 'rb') as fin:
#         data = json.load(fin)
#     read_prompts, QAs = [], []
#     for item in data:
#         if 'events' not in item:
#             continue
#         temp = []
#         for uttrance in item['events']:
#             if uttrance['action']=='message' and uttrance['data']!=None:
#                 if uttrance['agent']==0 and len(uttrance['data'].strip())!=0:
#                     temp.append({"role": "user", "content": uttrance['data'].strip()})
#                 elif uttrance['agent']==1 and len(uttrance['data'].strip())!=0:
#                     temp.append({"role": "assistant", "content": uttrance['data'].strip()})
#         read_prompts.append(temp)
#         buyer_price = item['scenario']['kbs'][0]['personal']['Target']
#         category = item['scenario']['category'] 
#         question = f"They were bargaining over a {category}. What is the price that the buyer had in mind?"
#         answer = f"The price that the buyer had in mind was {buyer_price} dollars."
#         QAs.append([
#                 {"role": "user", "content": question},
#                 {"role": "assistant", "content": answer}
#                 ])
#     def legal_conversation(conversation):
#         if len(conversation)<=4:
#             return False
#         pervious_role = conversation[0]['role']
#         for utterance in conversation[1:]:
#             if utterance['role']==pervious_role:
#                 return False
#             pervious_role = utterance['role']
#         return True
#     filtered_read_prompts, filtered_QA = [], []
#     for rp, qa in zip(read_prompts, QAs):
#         if legal_conversation(rp)==True:
#             filtered_read_prompts.append(rp)
#             filtered_QA.append(qa)
#     assert len(filtered_QA)==len(filtered_read_prompts)
#     return ToMLatentQADataset(
#         tokenizer,
#         filtered_read_prompts,
#         filtered_QA
#     )

# def get_CaSiNo_dataset(train_config, tokenizer, train=True):
#     data_path = train_config.train_qa if train else train_config.eval_qa
#     with open(data_path, 'rb') as fin:
#         data = json.load(fin)
#     read_prompts, QAs = [], []
#     for item in data:
#         temp = []
#         for uttrance in item['chat_logs']:
#             if uttrance['text'] in ['Submit-Deal', 'Accept-Deal']:
#                 continue
#             elif uttrance['id'] == 'mturk_agent_2':
#                 temp.append({"role": "user", "content": uttrance['text']})
#             elif uttrance['id'] == 'mturk_agent_1':
#                 temp.append({"role": "assistant", "content": uttrance['text']})
#         read_prompts.append(temp)
#         priorities, things = list(item['participant_info']['mturk_agent_2']['value2issue'].keys()), list(item['participant_info']['mturk_agent_2']['value2issue'].values())
#         question = "What is the priority of each item for the picnic?"
#         answer = f"The priority for {things[0]}, {things[1]} and {things[2]} are respectively {priorities[0]}, {priorities[1]} and {priorities[2]}."
#         QAs.append([
#                 {"role": "user", "content": question},
#                 {"role": "assistant", "content": answer}
#                 ])
#     assert len(QAs)==len(read_prompts)
#     return ToMLatentQADataset(
#         tokenizer,
#         read_prompts,
#         QAs
#     )

# def get_ToM_dataset(train_config, tokenizer, train=True):
#     if train_config.train_qa.find('CaSiNo')!=-1:
#         return get_CaSiNo_dataset(train_config, tokenizer, train)
#     elif train_config.train_qa.find('BARGAIN')!=-1:
#         return get_bargain_dataset(train_config, tokenizer, train)
    

def get_dataset(train_config, tokenizer, train=True):
    FILTER = train_config.filter.split("-")
    # print(FILTER)
    qa_path = train_config.train_qa if train else train_config.eval_qa
    # print(qa_path)
    with open(qa_path, "r") as f:
        qa_data = json.load(f)
        # print(len(qa_data), type(qa_data), list(qa_data.keys())[:10])
        NUM_QA = max([len(qa_data[label]) for label in qa_data])
        assert NUM_QA == min([len(qa_data[label]) for label in qa_data])
        # print(NUM_QA)

    def build_data_and_idx(path):
        # Get data
        data = defaultdict(list)
        if path == "":
            return data, []
        with open(path, "r") as f:
            raw_data = json.load(f)
            for item in raw_data:
                if item["label"].split("-")[0] in FILTER:
                    continue
                data[item["label"]].append(
                    (
                        item["control_user"],
                        item.get("control_model", ""),
                        item.get("stimulus_user", ""),
                        item.get("stimulus_model", ""),
                    )
                )
        # Get id tuples
        NUM_BEHAVIORS = max([len(data[label]) for label in data])
        # print(NUM_BEHAVIORS)
        assert NUM_BEHAVIORS == min([len(data[label]) for label in data])
        id_tuples = range(len(data) * NUM_BEHAVIORS * NUM_QA)
        # print(id_tuples)
        if train_config.train_percent == 1 or not train:
            id_tuples = list(id_tuples)
        else:
            id_tuples = random.sample(
                id_tuples, int(len(id_tuples) * train_config.train_percent)
            )
        for i in range(len(id_tuples)):
            label_idx = id_tuples[i] // (NUM_BEHAVIORS * NUM_QA)
            data_idx = (id_tuples[i] // NUM_QA) % NUM_BEHAVIORS
            qa_idx = id_tuples[i] % NUM_QA
            id_tuples[i] = (label_idx, data_idx, qa_idx)
        return data, id_tuples

    p1 = (
        train_config.train_stimulus_completion
        if train
        else train_config.eval_stimulus_completion
    )
    # print(p1)
    p2 = train_config.train_stimulus if train else train_config.eval_stimulus
    # print(p2)
    p3 = train_config.train_control if train else train_config.eval_control
    # print(p3)
    data_stimulus_completion = build_data_and_idx(p1)
    data_stimulus = build_data_and_idx(p2)
    data_control = build_data_and_idx(p3)
    # print(len(data_stimulus_completion))
    # sys.exit()

    return LatentQADataset(
        tokenizer,
        data_stimulus_completion,
        data_stimulus,
        data_control,
        qa_data,
    )


def get_dataloaders(train_config, tokenizer):
    dataset_train = get_ToM_dataset(train_config, tokenizer, train=True)
    # dataset_train = get_dataset(train_config, tokenizer, train=True)
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        collate_fn=DataCollatorForLatentQA(
            tokenizer,
            get_verb_mask=train_config.train_with_verb_mask,
            mask_all_but_last=False,
            nudge_persona=train_config.nudge_persona,
            modify_chat_template=train_config.modify_chat_template,
        ),
        batch_sampler=get_dist_batch_sampler(dataset_train, train_config, "train"),
    )
    if train_config.eval_ppl:
        # dataset_eval = get_dataset(train_config, tokenizer, train=False)
        dataset_eval = get_ToM_dataset(train_config, tokenizer, train=False)
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_eval,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            collate_fn=DataCollatorForLatentQA(
                tokenizer,
                get_verb_mask=train_config.train_with_verb_mask,
                mask_all_but_last=False,
                nudge_persona=train_config.nudge_persona,
                modify_chat_template=train_config.modify_chat_template,
            ),
            batch_sampler=get_dist_batch_sampler(dataset_eval, train_config, "val"),
        )
        return train_dataloader, eval_dataloader

    return train_dataloader, None
