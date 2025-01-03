import json
import pandas as pd
from torch.utils.data import Dataset
import random
import sys
from .dataset_utils import * 


class ToMLatentQASteerDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        read_prompts,
        QAs,
        is_dialog = True,
    ):
        self.BD = [
            {
                "role": "assistant",
                "content": "Sure, I've analyzed the dialog parties.",
            }
        ]
        self.tokenizer = tokenizer
        self.read_prompts = read_prompts
        self.QAs = QAs
        self.is_dialog = is_dialog
        self.lengths = []
        if self.is_dialog:
            for rp, qa in zip(read_prompts, QAs):
                self.lengths.append(sum([len(item['content']) for item in rp])+len(qa[0]['content'])+len(qa[1]['content']))
        else:
            for rp, qa in zip(read_prompts, QAs):
                self.lengths.append(len(rp)+len(qa[0]['content'])+len(qa[1]['content']))


    def __len__(self):
        return len(self.QAs)

    def __getitem__(self, idx):
        read_prompt = self.read_prompts[idx]
        qa_dialog = self.QAs[idx]
        # if self.is_dialog:
        read_prompt = self.tokenizer.apply_chat_template(
                read_prompt, tokenize=False, add_generation_prompt=True
            )
            
        # else:
        #     read_prompt = self.tokenizer.apply_chat_template(
        #             read_prompt, tokenize=False, add_generation_prompt=False
        #         )
        return {"read_prompt": read_prompt, "dialog": self.BD + qa_dialog}


def get_CaSiNo_dataset_steer(train_config, tokenizer, train=True):
    data_path = 'data/CaSiNo/test.json'
    with open(data_path, 'rb') as fin:
        data = json.load(fin)
    read_prompts, QAs = [], []
    for item in data:
        temp = []
        for uttrance in item['chat_logs']:
            if uttrance['text'] in ['Submit-Deal', 'Accept-Deal']:
                continue
            elif uttrance['id'] == 'mturk_agent_2':
                temp.append({"role": "assistant", "content": uttrance['text']})
            elif uttrance['id'] == 'mturk_agent_1':
                temp.append({"role": "user", "content": uttrance['text']})
        if temp[-1]['role'] == 'assistant':
            temp = temp[:-1]
        read_prompts.append(temp)
        
        question = "How much priority did each agent assign to different items?"
        answer = "For Agent 1: The priority for Food, Water and Firewood are respectively Low, Medium and High. For Agent 2: The priority for Food, Water and Firewood are respectively Low, Medium and High."
        QAs.append([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
                ])

    assert len(QAs)==len(read_prompts)
    return ToMLatentQASteerDataset(
        tokenizer,
        read_prompts,
        QAs
    )

def get_NegotiationToM_dataset_for_steer(train_config, tokenizer, train=True):
    data_path = 'data/NegotiationToM/test.json'
    with open(data_path, 'rb') as fin:
        data = json.load(fin)
    read_prompts, QAs = [], []
    for item in data:
        temp = []
        for uttrance in item['dialogue']:
            if uttrance.split(': ')[0] == 'agent_2':
                temp.append({"role": "assistant", "content": uttrance.split(': ')[1]})
            elif uttrance.split(': ')[0] == 'agent_1':
                temp.append({"role": "user", "content": uttrance.split(': ')[1]})
        read_prompts.append(temp)
        
        question = "What does the assistant believe about the priorities of different items for the user?"
        answer = f'The assistant thinks user asigns High priority to Water, Low priority to Food and Medium pririty to Firewood.'
        QAs.append([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
                ])

    assert len(QAs)==len(read_prompts)
    return ToMLatentQADataset(
        tokenizer,
        read_prompts,
        QAs,
        is_dialog=False
    )


def get_ToM_dataset_steer(steer_config, tokenizer, train=True):
    # if train_config.train_qa.find('CaSiNo')!=-1:
    return get_NegotiationToM_dataset_for_steer(steer_config, tokenizer, train)
    # return get_CaSiNo_dataset_steer(steer_config, tokenizer, train)
    # elif train_config.train_qa.find('BARGAIN')!=-1:
    #     return get_bargain_dataset(train_config, tokenizer, train)
    # elif train_config.train_qa.find('FANTOM')!=-1:
    #     return get_FanToM_dataset(train_config, tokenizer, train)
    # elif train_config.train_qa.find('NegotiationToM')!=-1:
    #     return get_NegotiationToM_dataset(train_config, tokenizer, train)
    # else:
    #     return Exception('There is NO such dataset!')
    


def get_steering_dataloaders(train_config, tokenizer):
    dataset_train = get_ToM_dataset_steer(train_config, tokenizer, train=True)
    # dataset_train = get_dataset(train_config, tokenizer, train=True)
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=1,
        pin_memory=True,
        collate_fn=DataCollatorForLatentQA(
            tokenizer,
            get_verb_mask=None,
            mask_all_but_last=True,
            nudge_persona=False,
            modify_chat_template=train_config.modify_chat_template,
        ),
        batch_sampler=get_batch_sampler(dataset_train, train_config, "train"),
    )

    return train_dataloader