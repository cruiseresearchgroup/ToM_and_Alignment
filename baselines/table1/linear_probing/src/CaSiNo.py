import pandas as pd
from copy import copy
from torch.utils.data import DataLoader
import os
import shutil
import argparse
import sys


# Fetching and Flattening Dataset
keys =  [
        'mturk_agent_1_value2issue_low', 'mturk_agent_1_value2issue_medium', 'mturk_agent_1_value2issue_high',
        'mturk_agent_1_value2reason_low', 'mturk_agent_1_value2reason_medium', 'mturk_agent_1_value2reason_high',
        'mturk_agent_1_outcomes_pointsscored', 'mturk_agent_1_outcomes_satisfaction', 'mturk_agent_1_outcomes_opponentlikeness',
        'mturk_agent_1_demographics_age', 'mturk_agent_1_demographics_gender', 'mturk_agent_1_demographics_ethnicity', 'mturk_agent_1_demographics_education',
        'mturk_agent_1_personality_svo',
        'mturk_agent_1_personality_bigfive_extraversion', 'mturk_agent_1_personality_bigfive_agreeableness', 'mturk_agent_1_personality_bigfive_conscientiousness', 'mturk_agent_1_personality_bigfive_emotionalstability', 'mturk_agent_1_personality_bigfive_opennesstoexperiences',
        'mturk_agent_2_value2issue_low', 'mturk_agent_2_value2issue_medium', 'mturk_agent_2_value2issue_high',
        'mturk_agent_2_value2reason_low', 'mturk_agent_2_value2reason_medium', 'mturk_agent_2_value2reason_high',
        'mturk_agent_2_outcomes_pointsscored', 'mturk_agent_2_outcomes_satisfaction', 'mturk_agent_2_outcomes_opponentlikeness',
        'mturk_agent_2_demographics_age', 'mturk_agent_2_demographics_gender', 'mturk_agent_2_demographics_ethnicity', 'mturk_agent_2_demographics_education',
        'mturk_agent_2_personality_svo',
        'mturk_agent_2_personality_bigfive_extraversion', 'mturk_agent_2_personality_bigfive_agreeableness', 'mturk_agent_2_personality_bigfive_conscientiousness', 'mturk_agent_2_personality_bigfive_emotionalstability', 'mturk_agent_2_personality_bigfive_opennesstoexperiences',
        'chat_log', 'final_decision'
        ]


def flatten_features_without_annotations(row):

    chat_log = row['chat_logs']
    participants_info = row['participant_info']
    mturk_agent_1_value2issue_low = participants_info['mturk_agent_1']['value2issue']['Low']
    mturk_agent_1_value2issue_medium = participants_info['mturk_agent_1']['value2issue']['Medium']
    mturk_agent_1_value2issue_high = participants_info['mturk_agent_1']['value2issue']['High']
    mturk_agent_1_value2reason_low = participants_info['mturk_agent_1']['value2reason']['Low']
    mturk_agent_1_value2reason_medium = participants_info['mturk_agent_1']['value2reason']['Medium']
    mturk_agent_1_value2reason_high = participants_info['mturk_agent_1']['value2reason']['High']
    mturk_agent_1_outcomes_pointsscored = participants_info['mturk_agent_1']['outcomes']['points_scored']
    mturk_agent_1_outcomes_satisfaction = participants_info['mturk_agent_1']['outcomes']['satisfaction']
    mturk_agent_1_outcomes_opponentlikeness = participants_info['mturk_agent_1']['outcomes']['opponent_likeness']
    mturk_agent_1_demographics_age = participants_info['mturk_agent_1']['demographics']['age']
    mturk_agent_1_demographics_gender = participants_info['mturk_agent_1']['demographics']['gender']
    mturk_agent_1_demographics_ethnicity = participants_info['mturk_agent_1']['demographics']['ethnicity']
    mturk_agent_1_demographics_education = participants_info['mturk_agent_1']['demographics']['education']
    mturk_agent_1_personality_svo = participants_info['mturk_agent_1']['personality']['svo']
    mturk_agent_1_personality_bigfive_extraversion = participants_info['mturk_agent_1']['personality']['big-five']['extraversion']
    mturk_agent_1_personality_bigfive_agreeableness = participants_info['mturk_agent_1']['personality']['big-five']['agreeableness']
    mturk_agent_1_personality_bigfive_conscientiousness = participants_info['mturk_agent_1']['personality']['big-five']['conscientiousness']
    mturk_agent_1_personality_bigfive_emotionalstability = participants_info['mturk_agent_1']['personality']['big-five']['emotional-stability']
    mturk_agent_1_personality_bigfive_opennesstoexperiences = participants_info['mturk_agent_1']['personality']['big-five']['openness-to-experiences']


    mturk_agent_1 = [mturk_agent_1_value2issue_low, mturk_agent_1_value2issue_medium, mturk_agent_1_value2issue_high,
                    mturk_agent_1_value2reason_low, mturk_agent_1_value2reason_medium, mturk_agent_1_value2reason_high,
                    mturk_agent_1_outcomes_pointsscored, mturk_agent_1_outcomes_satisfaction, mturk_agent_1_outcomes_opponentlikeness,
                    mturk_agent_1_demographics_age, mturk_agent_1_demographics_gender, mturk_agent_1_demographics_ethnicity, mturk_agent_1_demographics_education,
                    mturk_agent_1_personality_svo,
                    mturk_agent_1_personality_bigfive_extraversion, mturk_agent_1_personality_bigfive_agreeableness, mturk_agent_1_personality_bigfive_conscientiousness, mturk_agent_1_personality_bigfive_emotionalstability, mturk_agent_1_personality_bigfive_opennesstoexperiences
    ]

    participants_info = row['participant_info']
    mturk_agent_2_value2issue_low = participants_info['mturk_agent_2']['value2issue']['Low']
    mturk_agent_2_value2issue_medium = participants_info['mturk_agent_2']['value2issue']['Medium']
    mturk_agent_2_value2issue_high = participants_info['mturk_agent_2']['value2issue']['High']
    mturk_agent_2_value2reason_low = participants_info['mturk_agent_2']['value2reason']['Low']
    mturk_agent_2_value2reason_medium = participants_info['mturk_agent_2']['value2reason']['Medium']
    mturk_agent_2_value2reason_high = participants_info['mturk_agent_2']['value2reason']['High']
    mturk_agent_2_outcomes_pointsscored = participants_info['mturk_agent_2']['outcomes']['points_scored']
    mturk_agent_2_outcomes_satisfaction = participants_info['mturk_agent_2']['outcomes']['satisfaction']
    mturk_agent_2_outcomes_opponentlikeness = participants_info['mturk_agent_2']['outcomes']['opponent_likeness']
    mturk_agent_2_demographics_age = participants_info['mturk_agent_2']['demographics']['age']
    mturk_agent_2_demographics_gender = participants_info['mturk_agent_2']['demographics']['gender']
    mturk_agent_2_demographics_ethnicity = participants_info['mturk_agent_2']['demographics']['ethnicity']
    mturk_agent_2_demographics_education = participants_info['mturk_agent_2']['demographics']['education']
    mturk_agent_2_personality_svo = participants_info['mturk_agent_2']['personality']['svo']
    mturk_agent_2_personality_bigfive_extraversion = participants_info['mturk_agent_2']['personality']['big-five']['extraversion']
    mturk_agent_2_personality_bigfive_agreeableness = participants_info['mturk_agent_2']['personality']['big-five']['agreeableness']
    mturk_agent_2_personality_bigfive_conscientiousness = participants_info['mturk_agent_2']['personality']['big-five']['conscientiousness']
    mturk_agent_2_personality_bigfive_emotionalstability = participants_info['mturk_agent_2']['personality']['big-five']['emotional-stability']
    mturk_agent_2_personality_bigfive_opennesstoexperiences = participants_info['mturk_agent_2']['personality']['big-five']['openness-to-experiences']


    mturk_agent_2 = [mturk_agent_2_value2issue_low, mturk_agent_2_value2issue_medium, mturk_agent_2_value2issue_high,
                    mturk_agent_2_value2reason_low, mturk_agent_2_value2reason_medium, mturk_agent_2_value2reason_high,
                    mturk_agent_2_outcomes_pointsscored, mturk_agent_2_outcomes_satisfaction, mturk_agent_2_outcomes_opponentlikeness,
                    mturk_agent_2_demographics_age, mturk_agent_2_demographics_gender, mturk_agent_2_demographics_ethnicity, mturk_agent_2_demographics_education,
                    mturk_agent_2_personality_svo,
                    mturk_agent_2_personality_bigfive_extraversion, mturk_agent_2_personality_bigfive_agreeableness, mturk_agent_2_personality_bigfive_conscientiousness, mturk_agent_2_personality_bigfive_emotionalstability, mturk_agent_2_personality_bigfive_opennesstoexperiences
    ]

    final_decision = chat_log[-1]['text']

    return {k: v for k, v in zip(keys, mturk_agent_1+mturk_agent_2+[chat_log]+[final_decision])}

def flattening_helper_function(df):
    flattened_df = []
    for idx in range(len(df)):
        flattened_df.append(flatten_features_without_annotations(df.iloc[idx].to_dict()))
    return pd.DataFrame(flattened_df)

def incremental_chat(df):
    expanded_rows = []
    for _, row in df.iterrows():
        for idx in range(len(row['chat_log']))[2::2]:
            temp = copy(row)
            temp['chat_log'] = temp['chat_log'][:idx]
            temp['sequence'] = idx/2
            expanded_rows.append(temp)
    return pd.DataFrame(expanded_rows)

def fetch_and_prepare_dataset(url):
    df = pd.read_json(url)
    df = flattening_helper_function(df)
    df = incremental_chat(df).reset_index(drop=True)
    return df

# Changing format to suit Probing part
def reset_subdir(subdir):
    if os.path.exists(subdir):
        shutil.rmtree(subdir)
    os.makedirs(subdir)
    
def desire_m1(df, folder_name, length):
    reset_subdir(folder_name)
    labels_mapping = {
                    'Water_Food':'label1',
                    'Water_Firewood':'label2',
                    'Food_Water':'label3',
                    'Food_Firewood':'label4',
                    'Firewood_Water':'label5',
                    'Firewood_Food':'label6',
                }
    for idx, row in df[df.sequence>length].iterrows():
        temp_string = ''
        chat_hist = row['chat_log']
        for utt in chat_hist:
            if utt['id']=='mturk_agent_1':
                temp_string += f'### Human: {utt["text"]}\n\n'
            if utt['id']=='mturk_agent_2':
                temp_string += f'### Assistant: {utt["text"]}\n\n'
        label = labels_mapping[f'{row["mturk_agent_1_value2issue_low"]}_{row["mturk_agent_1_value2issue_medium"]}']
        with open(f'{folder_name}/conversation_{idx}_desirem1_{label}.txt', 'w') as fout:
            fout.write(temp_string.strip())

def belief_m1(df, folder_name, length):
    reset_subdir(folder_name)
    labels_mapping = {
                    'Water_Food':'label1',
                    'Water_Firewood':'label2',
                    'Food_Water':'label3',
                    'Food_Firewood':'label4',
                    'Firewood_Water':'label5',
                    'Firewood_Food':'label6',
                }
    for idx, row in df[df.sequence>length].iterrows():
        temp_string = ''
        chat_hist = row['chat_log']
        for utt in chat_hist:
            if utt['id']=='mturk_agent_1':
                temp_string += f'### Human: {utt["text"]}\n\n'
            if utt['id']=='mturk_agent_2':
                temp_string += f'### Assistant: {utt["text"]}\n\n'
        label = labels_mapping[f'{row["mturk_agent_2_value2issue_low"]}_{row["mturk_agent_2_value2issue_medium"]}']
        with open(f'{folder_name}/conversation_{idx}_beliefm1_{label}.txt', 'w') as fout:
            fout.write(temp_string.strip())

def desire_m2(df, folder_name, length):
    reset_subdir(folder_name)
    labels_mapping = {
                    'Water_Food':'label1',
                    'Water_Firewood':'label2',
                    'Food_Water':'label3',
                    'Food_Firewood':'label4',
                    'Firewood_Water':'label5',
                    'Firewood_Food':'label6',
                }
    for idx, row in df[df.sequence>length].iterrows():
        temp_string = ''
        chat_hist = row['chat_log']
        for utt in chat_hist:
            if utt['id']=='mturk_agent_2':
                temp_string += f'### Human: {utt["text"]}\n\n'
            if utt['id']=='mturk_agent_1':
                temp_string += f'### Assistant: {utt["text"]}\n\n'
        label = labels_mapping[f'{row["mturk_agent_2_value2issue_low"]}_{row["mturk_agent_2_value2issue_medium"]}']
        with open(f'{folder_name}/conversation_{idx}_desirem2_{label}.txt', 'w') as fout:
            fout.write(temp_string.strip())

def belief_m2(df, folder_name, length):
    reset_subdir(folder_name)
    labels_mapping = {
                    'Water_Food':'label1',
                    'Water_Firewood':'label2',
                    'Food_Water':'label3',
                    'Food_Firewood':'label4',
                    'Firewood_Water':'label5',
                    'Firewood_Food':'label6',
                }
    for idx, row in df[df.sequence>length].iterrows():
        temp_string = ''
        chat_hist = row['chat_log']
        for utt in chat_hist:
            if utt['id']=='mturk_agent_2':
                temp_string += f'### Human: {utt["text"]}\n\n'
            if utt['id']=='mturk_agent_1':
                temp_string += f'### Assistant: {utt["text"]}\n\n'
        label = labels_mapping[f'{row["mturk_agent_1_value2issue_low"]}_{row["mturk_agent_1_value2issue_medium"]}']
        with open(f'{folder_name}/conversation_{idx}_beliefm2_{label}.txt', 'w') as fout:
            fout.write(temp_string.strip())

# def belief_m1(df, folder_name, length):
#     reset_subdir(folder_name)
#     labels_mapping = {
#                     'Water_Food':'label1',
#                     'Water_Firewood':'label2',
#                     'Food_Water':'label3',
#                     'Food_Firewood':'label4',
#                     'Firewood_Water':'label5',
#                     'Firewood_Food':'label6',
#                 }
#     for idx, row in df[df.sequence>length].iterrows():
#         temp_string = ''
#         chat_hist = row['chat_log']
#         for utt in chat_hist:
#             if utt['id']=='mturk_agent_1':
#                 temp_string += f'### Human: {utt["text"]}\n\n'
#             if utt['id']=='mturk_agent_2':
#                 temp_string += f'### Assistant: {utt["text"]}\n\n'
#         label = labels_mapping[f'{row["mturk_agent_2_value2issue_low"]}_{row["mturk_agent_2_value2issue_medium"]}']
#         with open(f'{folder_name}/conversation_{idx}_beliefm1_{label}.txt', 'w') as fout:
#             fout.write(temp_string.strip())


# def prepare_label_data_m2(df, folder_name, length):
#     reset_subdir(folder_name)
#     labels_mapping = {
#                     'Water_Food':'label1',
#                     'Water_Firewood':'label2',
#                     'Food_Water':'label3',
#                     'Food_Firewood':'label4',
#                     'Firewood_Water':'label5',
#                     'Firewood_Food':'label6',
#                 }
#     for idx, row in df[df.sequence>length].iterrows():
#         temp_string = ''
#         chat_hist = row['chat_log']
#         for utt in chat_hist:
#             if utt['id']=='mturk_agent_1':
#                 temp_string += f'### Human: {utt["text"]}\n\n'
#             if utt['id']=='mturk_agent_2':
#                 temp_string += f'### Assistant: {utt["text"]}\n\n'
#         label = labels_mapping[f'{row["mturk_agent_2_value2issue_low"]}_{row["mturk_agent_2_value2issue_medium"]}']
#         with open(f'{folder_name}/conversation_{idx}_belief_{label}.txt', 'w') as fout:
#             fout.write(temp_string.strip())

# def prepare_label_data_m2(df, folder_name, length, label_column, labels_dict, name):
#     reset_subdir(folder_name)
#     # print(len(df[df.sequence>length]))
#     for idx, row in df[df.sequence>length].iterrows():
#         temp_string = ''
#         chat_hist = row['chat_log']
#         for utt in chat_hist:
#             if utt['id']=='mturk_agent_1':
#                 temp_string += f'### Assistant: {utt["text"]}\n\n'
#             if utt['id']=='mturk_agent_2':
#                 temp_string += f'### Human: {utt["text"]}\n\n'
#         with open(f'{folder_name}/conversation_{idx}_{name}_{labels_dict[f"{row[label_column]}"]}.txt', 'w') as fout:
#             fout.write(temp_string.strip())


if __name__ == '__main__':

    # train = fetch_and_prepare_dataset('https://raw.githubusercontent.com/kushalchawla/CaSiNo/refs/heads/main/data/split/casino_train.json')
    # valid = fetch_and_prepare_dataset('https://raw.githubusercontent.com/kushalchawla/CaSiNo/refs/heads/main/data/split/casino_valid.json')
    # test = fetch_and_prepare_dataset('https://raw.githubusercontent.com/kushalchawla/CaSiNo/refs/heads/main/data/split/casino_test.json')


    # prepare_label_data_m1(train, './dataset/desire_train', 4)
    # prepare_label_data_m1(valid, './dataset/desire_valid', 4)
    # prepare_label_data_m1(test, './dataset/desire_test', 4)

    # prepare_label_data_m2(train, './dataset/belief_train', 4)
    # prepare_label_data_m2(valid, './dataset/belief_valid', 4)
    # prepare_label_data_m2(test, './dataset/belief_test', 4)


    parser = argparse.ArgumentParser(description="Script to process train and test lengths with an optional argument.")
    parser.add_argument('train_length', type=int, help='Length of the training dataset')
    parser.add_argument('test_length', type=int, help='Length of the testing dataset')
    parser.add_argument('--flag', action='store_true', default=False, help='Optional flag, defaults to False')

    args = parser.parse_args()
    
    train_length = args.train_length
    test_length = args.test_length
    flag = args.flag

    if not flag:
        train = fetch_and_prepare_dataset('https://raw.githubusercontent.com/kushalchawla/CaSiNo/refs/heads/main/data/split/casino_train.json')
        # valid = fetch_and_prepare_dataset('https://raw.githubusercontent.com/kushalchawla/CaSiNo/refs/heads/main/data/split/casino_valid.json')
        # test = fetch_and_prepare_dataset('https://raw.githubusercontent.com/kushalchawla/CaSiNo/refs/heads/main/data/split/casino_test.json')

        desire_m1(train, './dataset/desirem1_train', train_length)
        desire_m2(train, './dataset/desirem2_train', train_length)
        # prepare_label_data_m1(test, './dataset/desire_test', test_length)

        belief_m1(train, './dataset/beliefm1_train', train_length)
        belief_m2(train, './dataset/beliefm2_train', train_length)
        # prepare_label_data_m2(test, './dataset/belief_test', test_length)

    if flag:
        test = fetch_and_prepare_dataset('https://raw.githubusercontent.com/kushalchawla/CaSiNo/refs/heads/main/data/split/casino_test.json')

        desire_m1(test, './dataset/desirem1_test', test_length)
        desire_m2(test, './dataset/desirem2_test', test_length)
        belief_m1(test, './dataset/beliefm1_test', test_length)
        belief_m2(test, './dataset/beliefm2_test', test_length)

        # prepare_label_data_m1(test, './dataset/waterhigh1_test', test_length, 'mturk_agent_1_value2issue_high', {'Water':'yes', 'Food':'no', 'Firewood':'no'}, 'waterhigh')
        # prepare_label_data_m2(test, './dataset/waterhigh2_test', test_length, 'mturk_agent_1_value2issue_high', {'Water':'yes', 'Food':'no', 'Firewood':'no'}, 'waterhigh')
        # prepare_label_data_m1(test, './dataset/foodhigh1_test', test_length, 'mturk_agent_1_value2issue_high', {'Water':'no', 'Food':'yes', 'Firewood':'no'}, 'foodhigh')
        # prepare_label_data_m2(test, './dataset/foodhigh2_test', test_length, 'mturk_agent_1_value2issue_high', {'Water':'no', 'Food':'yes', 'Firewood':'no'}, 'foodhigh')
        # prepare_label_data_m1(test, './dataset/firewoodhigh1_test', test_length, 'mturk_agent_1_value2issue_high', {'Water':'no', 'Food':'yes', 'Firewood':'no'}, 'firewoodhigh')
        # prepare_label_data_m2(test, './dataset/firewoodhigh2_test', test_length, 'mturk_agent_1_value2issue_high', {'Water':'no', 'Food':'no', 'Firewood':'yes'}, 'firewoodhigh')