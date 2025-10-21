from data.load_data import load_data 
from utils.config import POKEMON_TYPES
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
            p1_pokemon_state = t["p1_pokemon_state"]
            for key, val in p1_pokemon_state.items():
                if key == "effects": 
                    turn[f"p1_pokemon_state_{key}"] = val[0] # TODO: Think about solution 
                elif key == "boosts"  and isinstance(val, dict):
                    for boost_name, boost_stat in val.items(): 
                        turn[f"p1_pokemon_state_boost_{boost_name}"] = boost_stat
                else: 
                    turn[f"p1_pokemon_state_{key}"] = val
                         
            # 2. Flatten player 1 move details 
            p1_move_details = t["p1_move_details"]
            for key, val in p1_move_details.items():
                turn[f"p1_move_details_{key}"] = val
            
            # 3. Player 2 pokemon state: Flatten data and handle collections
            p2_pokemon_state = t["p2_pokemon_state"]
            for key, val in p2_pokemon_state.items():
                if key == "effects": 
                    turn[f"p2_pokemon_state_{key}"] = val[0] # TODO: Think about solution 
                elif key == "boosts" and isinstance(val, dict):
                    for boost_name, boost_stat in val.items(): 
                        turn[f"p2_pokemon_state_boost_{boost_name}"] = boost_stat
                else: 
                    turn[f"p2_pokemon_state_{key}"] = val

            # 4. Flatten player 2 move details 
            p2_move_details = t["p2_move_details"]
            for key, val in p2_move_details.items():
                turn[f"p2_move_details_{key}"] = val


            turns.append(turn)


    '''
    
    {'turn': 16, 
        'p1_pokemon_state': 
            {'name': 'gengar', 'hp_pct': 0.66, 'status': 'nostatus', 
                'effects': ['noeffect'], 
                'boosts': {'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0}}, 
        'p1_move_details': 
            {'name': 'thunderbolt', 'type': 'ELECTRIC', 'category': 'SPECIAL', 'base_power': 95, 'accuracy': 1.0, 'priority': 0}, 
        'p2_pokemon_state': 
            {'name': 'lapras', 'hp_pct': 0.37, 'status': 'nostatus', 'effects': ['noeffect'], 'boosts': {'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0}}, 
        'p2_move_details': 
            {'name': 'blizzard', 'type': 'ICE', 'category': 'SPECIAL', 'base_power': 120, 'accuracy': 0.9, 'priority': 0}
            }
    '''
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
     
    data = load_data()

    battles = extract_battles_df(data, train)
    turns = extract_turns_df(data)
    teams = extract_teams_df(data)

    # Somewhere, where we acutally clean data, we need to drop the flawed record row: 4877 
    # Not sure which one is actually the flawed record
    print(data[4877])
    print("PLACEHOLDER")

    return battles, turns, teams


 