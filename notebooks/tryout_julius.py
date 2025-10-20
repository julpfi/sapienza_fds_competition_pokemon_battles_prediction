from src.utils import config
import pandas
import json


train_data = []

# Read the file line by line
print(f"Loading data from '{config.DATA_TEST_PATH}'...")
try:
    with open(config.DATA_TEST_PATH, 'r') as f:
        for line in f:
            # json.loads() parses one line (one JSON object) into a Python dictionary
            train_data.append(json.loads(line))

    print(f"Successfully loaded {len(train_data)} battles.")

    # Let's inspect the first battle to see its structure
    print("\n--- Structure of the first train battle: ---")
    if train_data:
        first_battle = train_data[0]
        
        # To keep the output clean, we can create a copy and truncate the timeline
        battle_for_display = first_battle.copy()
        battle_for_display['battle_timeline'] = battle_for_display.get('battle_timeline', [])[:2] # Show first 2 turns
        
        # Use json.dumps for pretty-printing the dictionary
        print(json.dumps(battle_for_display, indent=4))
        if len(first_battle.get('battle_timeline', [])) > 3:
            print("    ...")
            print("    (battle_timeline has been truncated for display)")

        print(type(first_battle['p2_lead_details']['base_hp']))
        # One battle: dict of 
        #   'player_won' - boolean
        #   'p1_team_details' - list (team of pokemons) of dict of 
        #       'name' 
        #       'level'
        #       'types' - list of strings
        #       'base_hp' - all values are int
        #       'base_atk'
        #       'base_def'
        #       'base_spa'
        #       'base_spd'
        #       'base_spe'
        #   'p2_lead_details' - dict (one pokemon) of 
        #       'name'
        #       'level'
        #       'types' - list of strings
        #       'base_hp' 
        #       'base_atk'
        #       'base_def'
        #       'base_spa'
        #       'base_spd'
        #       'base_spe' 
        #   'battle_timeline' - list of dict of 
        #       'turn' 
        #       'p1_pokemon_state'
        #       'p1_move_details'   
        #       'p2_pokemon_state'
        #       'p2_move_details' 
        #   'battle_id'



except FileNotFoundError:
    print(f"ERROR: Could not find the training file at '{config.DATA_TRAIN_PATH}'.")
    print("Please make sure you have added the competition data to this notebook.")