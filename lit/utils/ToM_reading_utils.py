import json
import random
import pandas as pd 

def set_beliefQA_multiple_choices(qa):
    if qa['question_type'].endswith(":inaccessible"):
        option_a = qa['wrong_answer']
        option_b = qa['correct_answer']
    else:
        option_a = qa['wrong_answer']
        option_b = qa['correct_answer']

    answer_goes_last = random.choice([True, False])
    if answer_goes_last:
        choices = [option_a, option_b]
        answer = 1
    else:
        choices = [option_b, option_a]
        answer = 0

    # option letters iterate over the alphabet
    option_letters = ["(" + chr(x) + ")" for x in range(ord('a'), len(choices) + ord('a'))]
    choices_text = ""
    for letter, option in zip(option_letters, choices):
        choices_text += "{} {}\n".format(letter, option)

    return choices_text, answer

def setup_fantom(df, conversation_input_type='full'):
    total_num_q = 0
    for idx, _set in df.iterrows():
        total_num_q += len(_set['beliefQAs'])
        total_num_q += len(_set['answerabilityQAs_binary'])
        total_num_q += len(_set['infoAccessibilityQAs_binary'])
        if _set['factQA'] is not None:
            total_num_q += 1
        if _set['answerabilityQA_list'] is not None:
            total_num_q += 1
        if _set['infoAccessibilityQA_list'] is not None:
            total_num_q += 1

    inputs = []
    qas = []
    for idx, _set in df.iterrows():
        if conversation_input_type == "short":
            context = _set['short_context'].strip()
        elif conversation_input_type == "full":
            context = _set['full_context'].strip()
        
        set_id = _set['set_id']
        fact_q = _set['factQA']['question']
        fact_a = _set['factQA']['correct_answer']

        # Fact Question
        _set['factQA']['context'] = context
        input_text = "{}\n\nQuestion: {}\nAnswer:".format(context, fact_q)
        _set['factQA']['input_text'] = input_text
        _set['factQA']['set_id'] = set_id
        qas.append(_set['factQA'])
        inputs.append(input_text)

        for _belief_qa in _set['beliefQAs']:
            # Belief Questions
            _belief_qa['context'] = context
            input_text = "{}\n\nQuestion: {}\nAnswer:".format(context, _belief_qa['question'])
            _belief_qa['input_text'] = input_text
            _belief_qa['set_id'] = set_id
            qas.append(_belief_qa)
            inputs.append(input_text)

            # Multiple Choice Belief Questions
            _mc_belief_qa = {**_belief_qa}
            choices_text, answer = set_beliefQA_multiple_choices(_mc_belief_qa)
            mc_question = "{}\n{}\n\nChoose an answer from above:".format(_belief_qa['question'], choices_text.strip())
            _mc_belief_qa['question'] = mc_question
            _mc_belief_qa['question_type'] = _mc_belief_qa['question_type'] + ":multiple-choice"
            _mc_belief_qa['choices_text'] = choices_text
            _mc_belief_qa['choices_list'] = choices_text.strip().split("\n")
            _mc_belief_qa['correct_answer'] = answer
            input_text = "{}\n\nQuestion: {}".format(context, mc_question)
            _mc_belief_qa['input_text'] = input_text
            qas.append(_mc_belief_qa)
            inputs.append(input_text)

        # Answerability List Questions
        _set['answerabilityQA_list']['fact_question'] = fact_q
        _set['answerabilityQA_list']['context'] = context
        input_text = "{}\n\nTarget: {}\nQuestion: {}\nAnswer:".format(context, fact_q, _set['answerabilityQA_list']['question'])
        _set['answerabilityQA_list']['input_text'] = input_text
        _set['answerabilityQA_list']['set_id'] = set_id
        if conversation_input_type == "full" and len(_set['answerabilityQA_list']['wrong_answer']) > 0:
            _set['answerabilityQA_list']['missed_info_accessibility'] = 'inaccessible'
        qas.append(_set['answerabilityQA_list'])
        inputs.append(input_text)

        # Answerability Binary Questions
        if conversation_input_type == "full":
            missed_info_accessibility_for_full = _set['answerabilityQAs_binary'][0]['missed_info_accessibility']
            for _info_accessibility_qa in _set['answerabilityQAs_binary']:
                if _info_accessibility_qa['correct_answer'] != "yes":
                    missed_info_accessibility_for_full = 'inaccessible'

        for _answerability_qa in _set['answerabilityQAs_binary']:
            _answerability_qa['fact_question'] = fact_q
            _answerability_qa['context'] = context
            input_text = "{}\n\nTarget: {}\nQuestion: {} Answer yes or no.\nAnswer:".format(context, fact_q, _answerability_qa['question'])
            _answerability_qa['input_text'] = input_text
            _answerability_qa['set_id'] = set_id
            if conversation_input_type == "full":
                _answerability_qa['missed_info_accessibility'] = missed_info_accessibility_for_full
            qas.append(_answerability_qa)
            inputs.append(input_text)

        # Info Accessibility List Questions
        _set['infoAccessibilityQA_list']['fact_question'] = fact_q
        _set['infoAccessibilityQA_list']['fact_answer'] = fact_a
        _set['infoAccessibilityQA_list']['context'] = context
        input_text = "{}\n\nInformation: {} {}\nQuestion: {}\nAnswer:".format(context, fact_q, fact_a, _set['infoAccessibilityQA_list']['question'])
        _set['infoAccessibilityQA_list']['input_text'] = input_text
        _set['infoAccessibilityQA_list']['set_id'] = set_id
        if conversation_input_type == "full" and len(_set['infoAccessibilityQA_list']['wrong_answer']) > 0:
            _set['infoAccessibilityQA_list']['missed_info_accessibility'] = 'inaccessible'
        qas.append(_set['infoAccessibilityQA_list'])
        inputs.append(input_text)

        # Info Accessibility Binary Questions
        if conversation_input_type == "full":
            missed_info_accessibility_for_full = _set['infoAccessibilityQAs_binary'][0]['missed_info_accessibility']
            for _info_accessibility_qa in _set['infoAccessibilityQAs_binary']:
                if _info_accessibility_qa['correct_answer'] != "yes":
                    missed_info_accessibility_for_full = 'inaccessible'

        for _info_accessibility_qa in _set['infoAccessibilityQAs_binary']:
            _info_accessibility_qa['fact_question'] = fact_q
            _info_accessibility_qa['fact_answer'] = fact_a
            _info_accessibility_qa['context'] = context
            input_text = "{}\n\nInformation: {} {}\nQuestion: {} Answer yes or no.\nAnswer:".format(context, fact_q, fact_a, _info_accessibility_qa['question'])
            _info_accessibility_qa['input_text'] = input_text
            _info_accessibility_qa['set_id'] = set_id
            if conversation_input_type == "full":
                _info_accessibility_qa['missed_info_accessibility'] = missed_info_accessibility_for_full
            qas.append(_info_accessibility_qa)
            inputs.append(input_text)

    return inputs, qas

def get_bargain_text(train_config, tokenizer, train=True):
    data_path = train_config.train_qa if train else train_config.eval_qa
    with open(data_path, 'rb') as fin:
        data = json.load(fin)
    read_prompts, QAs = [], []
    for item in data:
        if 'events' not in item:
            continue
        temp = []
        for uttrance in item['events']:
            if uttrance['action']=='message' and uttrance['data']!=None:
                if uttrance['agent']==0 and len(uttrance['data'].strip())!=0:
                    temp.append({"role": "Buyer", "content": uttrance['data'].strip()})
                elif uttrance['agent']==1 and len(uttrance['data'].strip())!=0:
                    temp.append({"role": "Seller", "content": uttrance['data'].strip()})
        read_prompts.append(temp)

        buyer_price = item['scenario']['kbs'][0]['personal']['Target']
        seller_price = item['scenario']['kbs'][1]['personal']['Target']
        
        category = item['scenario']['category'] 
        question = f"They were bargaining over a {category}. What are the offered prices each party has in mind?"
        answer = f"The offered price of the seller is {seller_price}, and the buyer's is {buyer_price} dollars."
        QAs.append([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
                ])
        
    def legal_conversation(conversation):
        if len(conversation)<=4:
            return False
        pervious_role = conversation[0]['role']
        for utterance in conversation[1:]:
            if utterance['role']==pervious_role:
                return False
            pervious_role = utterance['role']
        return True
    filtered_read_prompts, filtered_QA = [], []
    for rp, qa in zip(read_prompts, QAs):
        if legal_conversation(rp)==True:
            filtered_read_prompts.append(rp)
            filtered_QA.append(qa)
    assert len(filtered_QA)==len(filtered_read_prompts)
    return filtered_read_prompts, filtered_QA

def get_Spoken_text(train_config, tokenizer, train=True):
    data_path = train_config.train_qa if train else train_config.eval_qa
    with open(data_path, 'rb') as fin:
        data = json.load(fin)
    read_prompts, QAs = [], []
    for item in data:
        length = sum([len(utt['content'].split(' ')) for utt in item['chat_log']])
        if length<600:
            read_prompts.append(item['chat_log'])
            question = "They were bargaining over a housing. What are the offered prices each party has in mind?"
            answer = f"The offered price of the seller is {item['seller_price']},000, and the buyer's is {item['buyer_price']},000 dollars."
            QAs.append([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                    ])
    return read_prompts, QAs


def get_CaSiNo_text(train_config, tokenizer, train=True):
    data_path = train_config.train_qa if train else train_config.eval_qa
    with open(data_path, 'rb') as fin:
        data = json.load(fin)
    read_prompts, QAs = [], []
    for item in data:
        temp = []
        for uttrance in item['chat_logs']:
            if uttrance['text'] in ['Submit-Deal', 'Accept-Deal']:
                continue
            elif uttrance['id'] == 'mturk_agent_2':
                temp.append({"role": "Agent 2", "content": uttrance['text']})
            elif uttrance['id'] == 'mturk_agent_1':
                temp.append({"role": "Agent 1", "content": uttrance['text']})
        read_prompts.append(temp)
        
        priorities_1, things_1 = list(item['participant_info']['mturk_agent_1']['value2issue'].keys()), list(item['participant_info']['mturk_agent_1']['value2issue'].values())
        priorities_2, things_2 = list(item['participant_info']['mturk_agent_2']['value2issue'].keys()), list(item['participant_info']['mturk_agent_2']['value2issue'].values())
        question = "How much priority did each agent assign to different items?"
        answer = f"For Agent 1: The priority for {things_1[0]}, {things_1[1]} and {things_1[2]} are respectively {priorities_1[0]}, {priorities_1[1]} and {priorities_1[2]}."
        answer += f" For Agent 2: The priority for {things_2[0]}, {things_2[1]} and {things_2[2]} are respectively {priorities_2[0]}, {priorities_2[1]} and {priorities_2[2]}."
        QAs.append([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
                ])

    assert len(QAs)==len(read_prompts)
    return read_prompts, QAs

def get_FanToM_text(train_config, tokenizer, train=True):
    data_path = train_config.train_qa if train else train_config.eval_qa
    df = pd.read_json(data_path)
    metas, QAs = setup_fantom(df)
    final_QAs, read_prompts = [], []
    for item, meta in zip(QAs, metas):
        final_QAs.append([
                {"role": "user", "content": item['input_text'].split('\n\n')[1].strip('\nAnswer:')},
                {"role": "assistant", "content": str(item['correct_answer'])}
        ])
        
        temp_read_prompt = []
        for line in item['input_text'].split('\n\n')[0].strip().split('\n'):
            temp_read_prompt.append({
                'role':line.split(':')[0].strip(),
                'content':line.split(':')[1].strip(),
            })
        read_prompts.append(temp_read_prompt)
    assert len(final_QAs)==len(read_prompts)
    return read_prompts, final_QAs

def get_JI_text(train_config, tokenizer, train=True):
    data_path = train_config.train_qa if train else train_config.eval_qa
    with open(data_path, 'rb') as fin:
        data = json.load(fin)
    read_prompts, QAs = [], []
    for item in data:
        recruiter_id = item['users'][0]['id'] if item['users'][0]['id']=='recruiter' else item['users'][1]['id']
        worker_id = item['users'][0]['id'] if item['users'][0]['id']=='worker' else item['users'][1]['id']
        temp = []
        for utterance in item['comments']:
            if utterance['user_id']==recruiter_id:
                temp.append({'role': 'assistant', 'content':utterance['body']})
            if utterance['user_id']==worker_id:
                temp.append({'role': 'user', 'content':utterance['body']})
        read_prompts.append(temp)

        question = "How much weight does the user assign to different factors for his next job?"
        worker_weights = item['users'][0]['utilities'] if item['users'][0]['id']=='worker' else item['users'][1]['utilities']
        worker_weights = [str((factor['name'], round(factor['weight'], 2))) for factor in worker_weights]
        answer = f"Factors and their weights are as follows: {' '.join(worker_weights)}"
        QAs.append([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
                ])
    def legal_conversation(conversation):
        if len(conversation)<=4:
            return False
        length = sum([len(utt['content'].split(' ')) for utt in conversation])
        if length>400:
            return False
        # pervious_role = conversation[0]['role']
        # for utterance in conversation[1:]:
        #     if utterance['role']==pervious_role:
        #         return False
        #     pervious_role = utterance['role']
        return True
    filtered_read_prompts, filtered_QA = [], []
    for rp, qa in zip(read_prompts, QAs):
        if legal_conversation(rp)==True:
            filtered_read_prompts.append(rp)
            filtered_QA.append(qa)
    assert len(filtered_QA)==len(filtered_read_prompts)
    return filtered_read_prompts, filtered_QA

def get_NegotiationToM_text(train_config, tokenizer, train=True):
    data_path = train_config.train_qa if train else train_config.eval_qa
    with open(data_path, 'rb') as fin:
        data = json.load(fin)
    print(data[0])
    read_prompts, QAs = [], []
    for item in data:
        temp = []
        for uttrance in item['dialogue']:
            if uttrance.split(': ')[0] == 'agent_2':
                temp.append({"role": "Agent 2", "content": uttrance.split(': ')[1]})
            elif uttrance.split(': ')[0] == 'agent_1':
                temp.append({"role": "Agent 1", "content": uttrance.split(': ')[1]})
        read_prompts.append(temp)
        
        question = "what is the intent of each agent for the last utterances? What are the beliefs and desires of each agent?"
        agent1_intent = item['utterance1_intent'] if item['utterance1_agent']=='agent_1' else item['utterance2_intent']
        agent2_intent = item['utterance1_intent'] if item['utterance1_agent']=='agent_2' else item['utterance2_intent']
        
        answer = f"The intent of the Agent 1 is [{agent1_intent}] and the intent of the Agent 2 is [{agent2_intent}]"
        answer += f' Regarding the Agent 1, Desire High: {item["agent1_desire_high"]}, Desire Medium: {item["agent1_desire_medium"]}, Desire Low: {item["agent1_desire_low"]},  Belief High: {item["agent1_belief_high"]}, Belief Medium: {item["agent1_belief_medium"]}, Belief Low: {item["agent1_belief_low"]}.' 
        answer += f' Regarding the Agent 2, Desire High: {item["agent2_desire_high"]}, Desire Medium: {item["agent2_desire_medium"]}, Desire Low: {item["agent2_desire_low"]},  Belief High: {item["agent2_belief_high"]}, Belief Medium: {item["agent2_belief_medium"]}, Belief Low: {item["agent2_belief_low"]}.' 
        QAs.append([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
                ])

    assert len(QAs)==len(read_prompts)
    return read_prompts, QAs

def batch_index_generator(dataset_size, batch_size):
    current_index = 0
    while current_index < dataset_size:
        end_index = min(current_index + batch_size, dataset_size)
        yield list(range(current_index, end_index))
        current_index = end_index

