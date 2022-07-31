import os
import pickle
import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
datasets.logging.set_verbosity_error()

label_per_class = 30
val_ratio = 0.2
num_unlabled_per_class = 50000000000000
num_val_per_class = 30
dataset_names = ["sst-2"]
trg_dir = f"ssl_data_30_30"
keys = {
    "elec": ["text", "label"],
    "ag_news": ["text", "label"],
    "imdb": ["text", "label"],
    "sst-2": ["sentence", "label"],
    "dbpedia_14": ["content", "label"],
    "yahoo_answers_topics": ["text", "topic"] #question_content
}

def df_to_list(df, dset_name):   
    df_new = pd.DataFrame()
    df_new["text"] = df[keys[dset_name][0]]
    df_new["label"] = df[keys[dset_name][1]]
    return df_new
    # return df


for dset_name in dataset_names:

    for i, seed in enumerate([0, 20, 50]):
        
        print(f"Dataset: {dset_name}, split {i}")
        dest_dir = f"{trg_dir}/split{i}/{dset_name}/"
        np.random.seed(seed)
        if not os.path.exists(dest_dir): 
            os.makedirs(dest_dir)

        if dset_name == 'elec':

            with open("ssl_data/elec_raw/elec_train.pkl", "rb") as f:
                train_df = pickle.load(f)
            with open("ssl_data/elec_raw/elec_test.pkl", "rb") as f:
                test_df = pickle.load(f)

            train_dset, val_dset = train_test_split(train_df, test_size=val_ratio, random_state=seed)
            test_dset = test_df
            unsupervised_dset = None

        elif dset_name == 'yahoo_answers_topics':

            with open("ssl_data/yahoo_raw/yahoo_train_clean.pkl", "rb") as f:
                train_df = pickle.load(f)
            with open("ssl_data/yahoo_raw/yahoo_test_clean.pkl", "rb") as f:
                test_df = pickle.load(f)

            train_dset, val_dset = train_test_split(train_df, test_size=val_ratio, random_state=seed)
            test_dset = test_df
            unsupervised_dset = None

        else:
            if dset_name != 'sst-2':
                whole_dset = datasets.load_dataset(dset_name)
            else:
                whole_dset = datasets.load_dataset('glue', 'sst2')

            train_dset = whole_dset['train'].train_test_split(val_ratio, seed=seed)
            train_dset, val_dset = train_dset['train'].to_pandas(), train_dset['test'].to_pandas()
            test_dset = whole_dset['test'].to_pandas()
            unsupervised_dset = whole_dset.get("unsupervised", None)
        # print("Len:", ", ".join([str(len(x)) for x in [train_dset, val_dset, test_dset]]))
        
        ''' Training data '''     
        ### Using all the data, instead of class-separating data
        if label_per_class < 1: 
            class_separated_data = train_dset    
            with open(f"{dest_dir}/train_all.pkl", "wb") as f:
                pickle.dump(df_to_list(class_separated_data, dset_name), f)
            
            continue

        else: 
            train_dset.reset_index(inplace=True)
            train_dset["index"] = train_dset.index
            def select_fn(df):
                selected_df = df.copy()
                selected_df.reindex(np.random.permutation(selected_df.index))
                return selected_df["index"][ :label_per_class]
            train_ids = train_dset.groupby(keys[dset_name][1]).apply(select_fn)
            class_separated_data = train_dset.iloc[train_ids.explode()] # explode is like "flatten"
            class_separated_data = class_separated_data.drop(columns="index") # remove helper

            # print("Class ids count:") 
            # print(class_separated_data['label'].value_counts())

            # random shuffle training data
            class_separated_data.reindex(np.random.permutation(class_separated_data.index))
            with open(f"{dest_dir}/train.pkl", "wb") as f:
                pickle.dump(df_to_list(class_separated_data, dset_name), f)

        

        ''' Unlabeled data ''' 
        train_dset = train_dset.drop(index=train_ids)
        if unsupervised_dset:
            unsupervised_dset = unsupervised_dset.to_pandas().append(train_dset)
        else:            
            unsupervised_dset = train_dset.copy()
            unsupervised_dset.reset_index(inplace=True)
            unsupervised_dset["index"] = unsupervised_dset.index
            def select_fn(df):
                selected_df = df.copy()
                selected_df.reindex(np.random.permutation(selected_df.index))
                size = min(num_unlabled_per_class, len(selected_df))
                return selected_df["index"]
            unsupervised_ids = unsupervised_dset.groupby(keys[dset_name][1]).apply(select_fn)
            unsupervised_dset = unsupervised_dset.iloc[unsupervised_ids.explode()] # explode is like "flatten"
            unsupervised_dset = unsupervised_dset.drop(columns="index") # remove helper

        with open(f"{dest_dir}/unsupervised.pkl", "wb") as f:
            pickle.dump(df_to_list(unsupervised_dset, dset_name), f)


        ''' Val/Test data '''  
        if dset_name == "sst-2":
            test_dset = val_dset.copy()
            print("[Warning] Use whole val dataset as test set for sst-2 dataset!!!")

        val_dset.reset_index(inplace=True)
        val_dset["index"] = val_dset.index
        def select_fn(df):
            selected_df = df.copy()
            selected_df.reindex(np.random.permutation(selected_df.index))
            return selected_df["index"][ :num_val_per_class]
        val_ids = val_dset.groupby(keys[dset_name][1]).apply(select_fn)
        val_dset = val_dset.iloc[val_ids.explode()] # explode is like "flatten"
        val_dset = val_dset.drop(columns="index") # remove helper

        with open(f"{dest_dir}/val.pkl", "wb") as f:
            pickle.dump(df_to_list(val_dset, dset_name), f)



        with open(f"{dest_dir}/test.pkl", "wb") as f:
            pickle.dump(df_to_list(test_dset, dset_name), f)
        
        