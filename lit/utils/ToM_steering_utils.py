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
    data_path = 'data/NegotiationToM/valid.json'
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
        
        question = "What is the intention of the assistant in the last utterance?"
        
        answer = f"The intent of the assistant is [Promote-Coordination]"
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


def get_NegotiationToM_dataset_for_steer_belief(train_config, tokenizer, train=True):
    data_path = 'data/NegotiationToM/valid.json'
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
        
        question = "What item does the assistant believe would have high priority for the user?"
        
        answer = f"The assistant believes Water would have a high probability for the user."
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
    # return get_NegotiationToM_dataset_for_steer(steer_config, tokenizer, train)
    return get_NegotiationToM_dataset_for_steer_belief(steer_config, tokenizer, train)
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

def get_evaluation_chats_NegotiationToM(intention):
    with open('data/NegotiationToM/test.json', 'r') as fin: 
        data = json.load(fin)
    chats, golden_response = [], []
    for item in data:
        if len(item['intents'])==len(item['dialogue']):
            for idx, intent in enumerate(item['intents']):
                temp = []
                if (intent.find(intention)!= -1) and (item['dialogue'][idx].startswith('agent_2')): 
                    chat_sample = item['dialogue'][:idx-1]
                    for uttrance in chat_sample:
                        if uttrance.split(': ')[0] == 'agent_2':
                            temp.append({"role": "assistant", "content": uttrance.split(': ')[1]})
                        elif uttrance.split(': ')[0] == 'agent_1':
                            temp.append({"role": "user", "content": uttrance.split(': ')[1]})
                    # chats.append(temp)
                    # golden_response.append(item['dialogue'][idx].split(': ')[1])
                elif (intent.find(intention)!= -1) and (item['dialogue'][idx].startswith('agent_1')): 
                    chat_sample = item['dialogue'][:idx-1]
                    for uttrance in chat_sample:
                        if uttrance.split(': ')[0] == 'agent_1':
                            temp.append({"role": "assistant", "content": uttrance.split(': ')[1]})
                        elif uttrance.split(': ')[0] == 'agent_2':
                            temp.append({"role": "user", "content": uttrance.split(': ')[1]})
                if temp!=[]:
                    chats.append(temp)
                    golden_response.append(item['dialogue'][idx].split(': ')[1])
    return chats, golden_response

def get_evaluation_chats_NegotiationToM_belief(high=None):
    with open('data/NegotiationToM/test.json', 'r') as fin: 
        data = json.load(fin)
    chats, golden_response = [], []
    for item in data:
        temp = []
        if (item['agent1_belief_high']=="Water") and (len(item['dialogue'])>2) and (item['dialogue'][-1].startswith('agent_1')):
            chat_sample = item['dialogue']
            for utterance in chat_sample[:-1]:
                if utterance.split(': ')[0] == 'agent_1':
                    temp.append({"role": "assistant", "content": utterance.split(': ')[1]})
                elif utterance.split(': ')[0] == 'agent_2':
                    temp.append({"role": "user", "content": utterance.split(': ')[1]})
            chats.append(temp)
            golden_response.append(chat_sample[-1].split(': ')[1])
        elif (item['agent2_belief_high']=="Water") and (len(item['dialogue'])>2) and (item['dialogue'][-1].startswith('agent_2')):
            chat_sample = item['dialogue']
            for utterance in chat_sample[:-1]:
                if utterance.split(': ')[0] == 'agent_2':
                    temp.append({"role": "assistant", "content": utterance.split(': ')[1]})
                elif utterance.split(': ')[0] == 'agent_1':
                    temp.append({"role": "user", "content": utterance.split(': ')[1]})
            chats.append(temp)
            golden_response.append(chat_sample[-1].split(': ')[1])
    return chats, golden_response