from src.utils import config
import json
import pandas as pd


def load_raw_data(train) -> list:
    path = config.DATA_TRAIN_PATH if train else config.DATA_TEST_PATH 
    print(f"Loading raw data from '{path}'...")
    train_data = []
    try:
        with open(path, 'r') as f:
            for line in f:
                train_data.append(json.loads(line))
    except FileNotFoundError:
        print(f"ERROR: Could not find the training file at '{path}'.")
    return train_data

def load_data(train: bool=True):
    '''
    Wrapper: Loads the raw data from the folder /data/raw 
    Param:
        train: Boolean - Selectes either the training data for training the model 
            or the test data for predicting the results and submitting   
    '''
    data_raw = load_raw_data(train)
    return data_raw