import pandas as pd
from .feature_selection import select_features

from .features_version_6 import feature_engineering_version_6
from .features_version_7 import feature_engineering_version_7
from .features_version_9 import feature_engineering_version_9
from .features_version_10 import feature_engineering_version_10
from .features_version_11 import feature_engineering_version_11
from .features_version_12 import feature_engineering_version_12


# try if works: from src.data.feature_engineering.features_version_1 import feature_engineering_version_1

def feature_engineering(battles:pd.DataFrame, turns:pd.DataFrame, teams:pd.DataFrame, version: int=1, train:bool=True):
   
    # Safety handling ensuring that train data contains player_won
    if train and "player_won" not in battles.columns:
        raise KeyError("Expected player_won column in train data but was not found")

    match version:       
        case 6: 
            features = feature_engineering_version_6(train, battles, turns, teams)    
            features = select_features(
                features, variance_threshold=0.01, correlation_threshold=0.95, 
                exclude_cols=['battle_id', 'player_won'] if train else ['battle_id'])    
        case 7: 
            features = feature_engineering_version_7(train, battles, turns, teams)
        case 9: 
            features = feature_engineering_version_9(train, battles, turns, teams)
        case 10:
            features = feature_engineering_version_10(train, battles, turns, teams)
        case 11:
            features = feature_engineering_version_11(train, battles, turns, teams)
        case 12:
            features = feature_engineering_version_12(train, battles, turns, teams)
        case _: 
            raise Exception("ERROR: Invalid selection of which set of features to use. \n -> feature_engineering.py")
        
    print("\nFinal features:")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(features.columns.tolist(), "\n\n")
    return features
