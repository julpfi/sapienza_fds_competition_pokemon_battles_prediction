from src.utils import config
import json

def load_data(train: bool=True) -> list:
    '''
    Loads the raw data from the folder /data/raw 
    Param:
        train: Boolean - Selectes either the training data for training the model 
            or the test data for predicting the results and submitting   
    Return: 
        list of batttles; each battle is a strcuture of dicts and list containing the information 
    '''

    path = config.DATA_TRAIN_PATH if train else config.DATA_TEST_PATH 
    print(f"Loading raw data from '{path}'...")
    data = []
    try:
        with open(path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"ERROR: Could not find the training file at '{path}'.")
    return data
