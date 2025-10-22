from load_data import load_data 
from src.utils.config import POKEMON_TYPES
import pandas as pd


def extract_battles_df(raw_data:list, train: bool=True) -> pd.DataFrame: 
    '''
    Description: 
        Extracts each battles player two descriptions and if training set the player_won column (primary key: battle_id)
        The method collects all the relevant stats for the pokemon on player 2 and flattens the types
    Param: 
        list: raw_data that contains the read battles 
    Return: 
        Pandas Dataframe with the described records
    '''
    battles = []

    for x in raw_data:
        battle = {"battle_id": x["battle_id"]}
        battle["player_won"] = x["player_won"] if train and "player_won" in x else None

        for key, val in x["p2_lead_details"].items():
            if key == "types" and isinstance(val, list):
                for t in POKEMON_TYPES:
                    battle[f"p2_lead_type_{t}"] = int(t in val)
            else:
                battle[f"p2_lead_{key}"] = val

        battles.append(battle)

    return pd.DataFrame(battles)


def extract_turns_df(raw_data: list) -> pd.DataFrame: 
    '''
    Description: 
        Extracts the turns of each battles (primary key: battle_id, turn)
        This method flattens all the pokemon states and move details of player one and two plus the boosts in the pokemon states
    Param: 
        list: raw_data that contains the read battles 
    Return: 
        Pandas Dataframe with the described records
    '''
    turns = []
    # Loops over all the battles in the raw data 
    for x in raw_data:
        battle_id = x["battle_id"]
        
        # Loops over every turn in each battle
        for t in x["battle_timeline"]:
            turn = {
                "battle_id":  battle_id, 
                "turn":  t["turn"]
            }

            # IDEA: Could split turns into p1 and p2? 

            # 1. Player 1 pokemon state: Flatten data and handle collections
            p1_pokemon_state = t.get("p1_pokemon_state") or {}
            for key, val in p1_pokemon_state.items():
                if key == "effects": 
                    turn[f"p1_pokemon_state_{key}"] = "_".join(val) # Concats the potential effects: Check if needed 
                elif key == "boosts"  and isinstance(val, dict):
                    for boost_name, boost_stat in val.items(): 
                        turn[f"p1_pokemon_state_boost_{boost_name}"] = boost_stat
                else: 
                    turn[f"p1_pokemon_state_{key}"] = val
                         
            # 2. Flatten player 1 move details 
            p1_move_details = t.get("p1_move_details") or {}
            for key, val in p1_move_details.items():
                if key == "type": 
                    turn[f"p1_move_details_{key}"] = str.lower(val)
                else: 
                    turn[f"p1_move_details_{key}"] = val
            
            # 3. Player 2 pokemon state: Flatten data and handle collections
            p2_pokemon_state = t.get("p2_pokemon_state") or {}
            for key, val in p2_pokemon_state.items():
                if key == "effects": 
                    turn[f"p2_pokemon_state_{key}"] =  "_".join(val) # Concats the potential effects: Check if needed
                elif key == "boosts" and isinstance(val, dict):
                    for boost_name, boost_stat in val.items(): 
                        turn[f"p2_pokemon_state_boost_{boost_name}"] = boost_stat 
                else: 
                    turn[f"p2_pokemon_state_{key}"] = val

            # 4. Flatten player 2 move details 
            p2_move_details = t.get("p2_move_details") or {}
            for key, val in p2_move_details.items():
                if key == "type": 
                    turn[f"p2_move_details_{key}"] = str.lower(val)
                else: 
                    turn[f"p2_move_details_{key}"] = val

            turns.append(turn)

    return pd.DataFrame(turns)


def extract_teams_df(raw_data: list) -> pd.DataFrame: 
    '''
    Description: 
        Extracts the team of player 1 (up to 6 pokemons) where each pokemon is a line (primary keys: battle_id, nr)
        The method collects all the relevant stats for the pokemons and flattens the types
    Param: 
        list: raw_data that contains the read battles 
    Return: 
        Pandas Dataframe with the described records
    '''
    team_pokemons = []
    for x in raw_data:
        battle_id = x["battle_id"]
        for nr, pokemon in enumerate(x["p1_team_details"]):
            team_pokemon = {
                "battle_id": battle_id,
                "pokemon_nr": nr,
            }

            for key, val in pokemon.items():
                if key == "types" and isinstance(val, list):
                    for t in POKEMON_TYPES:
                        team_pokemon[f"type_{t}"] = int(t in val)
                else:
                    team_pokemon[key] = val

            team_pokemons.append(team_pokemon)

    return pd.DataFrame(team_pokemons)


def clean_battles_df(df: pd.DataFrame) -> pd.DataFrame:
    return df


def clean_turns_df(df: pd.DataFrame) -> pd.DataFrame:
    return df


def clean_teams_df(df: pd.DataFrame) -> pd.DataFrame:
    return df


def clean_data(train: bool=True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load, extracts, and cleans the pokemon data
    Param: 
        Train: bool: Indicates if it is handling the test or train data and adjusts logic (mainly player_won column)
    Returns:
        battles_df: pandas df: One row per battle
        turns_df: pandas df: One row per turn per battle
        teams_df: pandas df: One row per pokemon per battle
    """
     
    # Load
    data = load_data()

    # Extract 
    battles = extract_battles_df(data, train)
    turns = extract_turns_df(data)
    teams = extract_teams_df(data)

    # Clean
    battles = clean_battles_df(battles)
    turns = clean_turns_df(turns)
    teams = clean_teams_df(teams)

    # Somewhere, where we acutally clean data, we need to drop the flawed record row: 4877 
    # Not sure which one is actually the flawed record
    # print(data[4877])
    print("PLACEHOLDER")
    
    return battles, turns, teams

'''
battles, turns, teams = clean_data()

print("BATTLES:")
print(battles.head())
print(battles.info())
print(battles.describe())
print(battles.nunique())
print(battles.isnull().sum())

print("TURNS:")
print(turns.head())
print("TEAMS:")
print(teams.head())
 
print(turns.head())

print(turns.info())
print()
print(turns.describe())
print()
print(turns.nunique())
print()
print(turns.isnull().sum())
'''

