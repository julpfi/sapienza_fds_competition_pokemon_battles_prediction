from src.utils import config
import json
import pandas as pd


def load_raw_data() -> list:
    print(f"Loading raw data from '{config.DATA_TRAIN_PATH}'...")
    train_data = []
    try:
        with open(config.DATA_TRAIN_PATH, 'r') as f:
            for line in f:
                train_data.append(json.loads(line))
    except FileNotFoundError:
        print(f"ERROR: Could not find the training file at '{config.DATA_TRAIN_PATH}'.")
    return train_data

def load_data():
    data_raw = load_raw_data()
    return data_raw