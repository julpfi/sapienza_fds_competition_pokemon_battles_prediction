from src.data.load_data import load_data 
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
    # Note: Checked that every p1 pokemon team has 6 pokemons -> It could be merged easily in battles_df
    # For now: Keep this structure as it is given and allows for flexibility when calculating teams specific features 
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


def clean_battles_df(df: pd.DataFrame, train:bool) -> pd.DataFrame:
    # Drop record as stated in Classroom 
    if train:
        df = df[df['battle_id'] != 4877].copy()

    # Cleans the battles_df. (Mainly p2 Lead data)
    # Based on our EDA, this df is mostly clean.
    # Just make sure 'player_won' is an integer if it exists.
    if train:
        df['player_won'] = df['player_won'].astype(int)
        
    return df


def clean_turns_df(df: pd.DataFrame, train:bool) -> pd.DataFrame:
    # Drop record as stated in Classroom 
    if train:
        df = df[df['battle_id'] != 4877].copy()

    # Identify all columns related to move details
    str_move_cols = [col for col in df.columns if col.endswith(('_name', '_type', '_category'))]
    num_move_cols = [col for col in df.columns if col.endswith(('_base_power', '_accuracy', '_priority'))]
    
    # 1. Fill NaNs for missing moves
    # When a move is missing (NaN), it's usually a switch or ko 
    
    # Fill text-based move columns with a 'MISSING' placeholder
    for col in str_move_cols:
        # Use .astype('category') to save a lot of memory
        df[col] = df[col].fillna('MISSING_MOVE').astype('category')
        
    # Fill numerical move columns with 0
    for col in num_move_cols:
        df[col] = df[col].fillna(0.0)

    # 2. Optimize data types
    boost_cols = [col for col in df.columns if 'pokemon_state_boost_' in col]
    for col in boost_cols:
        df[col] = df[col].astype('int8')
        
    return df


def clean_teams_df(df: pd.DataFrame, train:bool) -> pd.DataFrame:
    #Cleans the teams_df. (p1 Team data)
    # EDA showed this was already clean. No action needed for now.

    # Drop record as stated in Classroom 
    if train:
        df = df[df['battle_id'] != 4877].copy()
    return df


def clean_data(raw_data: list, train: bool=True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extracts and cleans the pokemon data
    Param: 
        raw_data: list: Raw_data that is loaded in load_data
        Train: bool: Indicates if it is handling the test or train data and adjusts logic (mainly player_won column)
    Returns:
        battles_df: pandas df: One row per battle
        turns_df: pandas df: One row per turn per battle
        teams_df: pandas df: One row per pokemon per battle
    """
     
    # Extract 
    print("Extracting battles, turns, and teams dataframes...")
    battles = extract_battles_df(raw_data, train)
    turns = extract_turns_df(raw_data)
    teams = extract_teams_df(raw_data)

    # Clean
    print("Cleaning battles, turns, and teams dataframes...\n")
    battles = clean_battles_df(battles, train)
    turns = clean_turns_df(turns, train)
    teams = clean_teams_df(teams, train)
    
    return battles, turns, teams

