from clean_data import clean_data
import pandas as pd


def feature_engineering_version_1(train:bool, battles:pd.DataFrame, turns:pd.DataFrame, teams:pd.DataFrame): 
    
    p1_mean_stats = (
        teams.groupby("battle_id")
        .agg(
            mean_p1_hp=("base_hp", "mean"),
            mean_p1_atk=("base_atk", "mean"),
            mean_p1_def=("base_def", "mean"),
            mean_p1_spa=("base_spa", "mean"),
            mean_p1_spd=("base_spd", "mean"),
            mean_p1_spe=("base_spe", "mean"),
            mean_p1_level=("level", "mean"),
        )
        .reset_index() # moves the merge key (battle_id) back as a column that can be used later
    )
    
    # Takes the columns names of p2 lead, battle_id and player_won (if train) that should be used for selection 
    columns_battles = (
        [col for col in battles.columns if col.startswith("p2_lead_")]
        + ["battle_id"]
        + (["player_won"] if train else [])
        )

    battles_subset = battles[columns_battles]


    # use merge (not join) in order to join two dfs based on the foreing key 
    df = battles_subset.merge(p1_mean_stats, on="battle_id", how="left")

    # selection of stats that we want a difference on: 
    for stat in ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe", "level"]:
        p2_col = f"p2_lead_{stat}"
        p1_col = f"mean_p1_{stat}"
        diff_col = f"diff_{stat}"
        if p2_col in df.columns and p1_col in df.columns:
            df[diff_col] = df[p1_col] - df[p2_col]

    return df


def feature_engineering(version: int=1, train:bool=True):
    battles, turns, teams = clean_data(train)
   
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
    
    
X = feature_engineering(train=False)

print(X)
#print(y)
