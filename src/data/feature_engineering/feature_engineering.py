import pandas as pd
from features_version_1 import feature_engineering_version_1

def feature_engineering(battles:pd.DataFrame, turns:pd.DataFrame, teams:pd.DataFrame, version: int=1, train:bool=True):
   
    # Safety handling ensuring that train data contains player_won
    if train and "player_won" not in battles.columns:
        raise KeyError("Expected player_won column in train data but was not found")

    match version: 
        case 1: 
            features = feature_engineering_version_1(train, battles, turns, teams)
        case _: 
            raise Exception("ERROR: Invalid selection of which set of features to use. \n -> feature_engineering.py")
        
     # train data will conatin "player_won" column; test does not -> returns accordingly 
    if train:
        y = features["player_won"]
        X = features.drop(columns=["player_won"])
        return X, y
    else:
        return features
