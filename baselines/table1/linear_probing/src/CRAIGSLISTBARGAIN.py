import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def flatten_features(row):
    category = row['scenario']['category'] 

    buyer_price = row['scenario']['kbs'][0]['personal']['Target']
    buyer_bottomline = row['scenario']['kbs'][0]['personal']['Bottomline']
    buyer_description = row['scenario']['kbs'][0]['item']['Description']
    
    seller_price = row['scenario']['kbs'][1]['personal']['Target']
    seller_bottomline = row['scenario']['kbs'][1]['personal']['Bottomline']
    seller_description = row['scenario']['kbs'][1]['item']['Description']

    chat_log = row['events']

    sample = {
        'category':category,
        'chat_log':chat_log,
        'buyer_price':buyer_price,
        'buyer_bottomline':buyer_bottomline,
        'buyer_description':buyer_description,
        'seller_price':seller_price,
        'seller_bottomline':seller_bottomline,
        'seller_description':seller_description,
        
    }

    return sample

def flattening_helper_function(df):
    flattened_df = []
    for idx in range(len(df)):
        flattened_df.append(flatten_features(df.iloc[idx].to_dict()))
    return pd.DataFrame(flattened_df)

def fetch_and_prepare_dataset(url):
    df = pd.read_json(url)
    df = flattening_helper_function(df)
    df['buyer_price'] = df.apply(lambda row: row['buyer_price'], axis=1)
    df['seller_price'] = df.apply(lambda row: row['seller_price'], axis=1)
    df = df[['category', 'chat_log', 'buyer_price', 'seller_price']]
    
    return df


def dialogue_merge(row, exclude=''):
        category = row['category']
        c_log = row['chat_log']
        dialog = []
        final_dialog_acts = []
        for idx, utter in enumerate(c_log):
            if 'metadata' in utter:
                if (utter["action"]=='message') and (utter['metadata']!=None): 
                    if exclude!=utter['metadata']['intent']:    
                        dialog.append(f'Agent {utter["agent"]+1}: {utter["data"]}')
                        final_dialog_acts.append(utter['metadata']['intent'])
        assert exclude not in final_dialog_acts
        return f'Bargaining over a {category} | ' + ' | '.join(dialog)

def causal_dialogue_merge(row, exclude=''):
        category = row['category']
        c_log = row['chat_log']
        dialog = []
        final_dialog_acts = []
        for idx, utter in enumerate(c_log):
            if 'metadata' in utter:
                if (utter["action"]=='message') and (utter['metadata']!=None): 
                    if exclude!=utter['metadata']['intent']:    
                        dialog.append(f'Agent {utter["agent"]+1}: {utter["data"]}')
                        final_dialog_acts.append(utter['metadata']['intent'])
        assert exclude not in final_dialog_acts
        return f'Buyer and seller each have a price tag in their mind and start bargaining on a {category} | ' + ' | '.join(dialog).replace('Agent 1:', 'buyer:').replace('Agent2:', 'seller:')

def get_dialog_acts_count(df):
     all_acts = []
     for idx, row in df.iterrows():
          for utter in row['chat_log']:
               if 'metadata' in utter:
                    if utter["action"]=='message':
                        if utter['metadata']!=None and 'intent' in utter['metadata']:
                            all_acts.append(utter['metadata']['intent'])
     return dict(Counter(all_acts))

def extract_dialogue_statistics(dialogues):
    a1_utter_count, a2_utter_count = '<>'.join(dialogues).count('Agent 1:'), '<>'.join(dialogues).count('Agent 2:')
    word_frequency = dict(Counter('<>'.join(dialogues).replace('Agent 1:', '').replace('Agent 2:', '').replace(' | ', '').split(' ')).most_common(25))
    word_frequency.pop('')
    return a1_utter_count, a2_utter_count, word_frequency

def get_changed_dialogs_count(df, dialog_act):
    def has_dialog_act(items, dialog_act):
        for item in items:
            if ('metadata' in item) and (item['metadata']!=None):
                if item['metadata']['intent']==dialog_act:
                    return True
        return False
    index = df['chat_log'].apply(lambda items:has_dialog_act(items, dialog_act))
    return len(df[index])



if __name__=='__main__':
    train = fetch_and_prepare_dataset('../datasets/CRAIGSLISTBARGAIN/train.json')


