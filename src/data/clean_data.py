from src.utils import config
from data.load_data import load_data 
import json
import pandas as pd



def clean_data():
    # Executes all necessary functins from above

    train_data = load_data()

    # Somewhere, where we acutally clean data, we need to drop the flawed record row: 4877 
    # Not sure which one is actually the flawed record
    print(train_data[4877])
    print("PLACEHOLDER")


 