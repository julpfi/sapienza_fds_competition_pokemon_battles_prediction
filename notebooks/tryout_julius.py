from src.utils import config
import pandas
import json


train_data = []
test_data = []
types = set()


try:
    with open(config.DATA_TRAIN_PATH, 'r') as f:
        for line in f:
            # json.loads() parses one line (one JSON object) into a Python dictionary
            train_data.append(json.loads(line))
        
        
    print(f"Successfully loaded {len(train_data)} battles.")
    with open(config.DATA_TEST_PATH, 'r') as f:
        for line in f:
            # json.loads() parses one line (one JSON object) into a Python dictionary
            test_data.append(json.loads(line))
    print(f"Successfully loaded {len(test_data)} battles.") 




    # ------------------ CHECK AND QUERY TYPES -------------------------------
    for l in train_data:
        for t1 in l["p1_team_details"]:
            types.update([t.lower() for t in t1["types"]])
            
        types.update([t.lower() for t in l["p2_lead_details"]["types"]])


    for l in test_data:
        for t1 in l["p1_team_details"]:
            types.update([t.lower() for t in t1["types"]])
            
        types.update([t.lower() for t in l["p2_lead_details"]["types"]])

    for b in train_data:
        for t in b["battle_timeline"]:
            if t["p1_move_details"]:
                types.add(t["p1_move_details"]["type"].lower())
            if t["p2_move_details"]:
                types.add(t["p2_move_details"]["type"].lower())

    for b in test_data:
        for t in b["battle_timeline"]:
            if t["p1_move_details"]:
                types.add(t["p1_move_details"]["type"].lower())
            if t["p2_move_details"]:
                types.add(t["p2_move_details"]["type"].lower())

    # print("TYPES:\n", types)
    

    # -------------------------------------------------------------------------------


    # ------------------ CHECK NUMBER OF POKEMON OF PLAYER 1  -------------------------------

    team_sizes_train = [len(battle["p1_team_details"]) for battle in train_data]
    team_sizes_test = [len(battle["p1_team_details"]) for battle in test_data]
    
    #print(set(team_sizes_train))
    #print(set(team_sizes_test))

    # ---------------------------------------------------------------------------------

    # Let's inspect the first battle to see its structure
    print("\n--- Structure of the first train battle: ---")
    if train_data:
        first_battle = train_data[0]
        
        # To keep the output clean, we can create a copy and truncate the timeline
        battle_for_display = first_battle.copy()
        battle_for_display['battle_timeline'] = battle_for_display.get('battle_timeline', [])[:2] # Show first 2 turns
        
        # Use json.dumps for pretty-printing the dictionary
        
        '''print(json.dumps(battle_for_display, indent=4))
        if len(first_battle.get('battle_timeline', [])) > 3:
            print("    ...")
            print("    (battle_timeline has been truncated for display)")

        '''


        # print(first_battle['battle_timeline'][15])
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


        '''
        {'turn': 2, 
        'p1_pokemon_state': 
            {'name': 'chansey', 'hp_pct': 1.0, 'status': 'par', 'effects': ['noeffect'], 
                'boosts': {'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0}}, 
        'p1_move_details': None, 
        'p2_pokemon_state': 
            {'name': 'chansey', 'hp_pct': 1.0, 'status': 'nostatus', 'effects': ['noeffect'], 
                'boosts': {'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0}}, 
        'p2_move_details': None}
        


        {'turn': 16, 
        'p1_pokemon_state': 
            {'name': 'gengar', 'hp_pct': 0.66, 'status': 'nostatus', 'effects': ['noeffect'], 'boosts': {'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0}}, 
        'p1_move_details': 
            {'name': 'thunderbolt', 'type': 'ELECTRIC', 'category': 'SPECIAL', 'base_power': 95, 'accuracy': 1.0, 'priority': 0}, 
        'p2_pokemon_state': 
            {'name': 'lapras', 'hp_pct': 0.37, 'status': 'nostatus', 'effects': ['noeffect'], 'boosts': {'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0}}, 
        'p2_move_details': 
            {'name': 'blizzard', 'type': 'ICE', 'category': 'SPECIAL', 'base_power': 120, 'accuracy': 0.9, 'priority': 0}}
        
        

        '''

    df_train = pandas.DataFrame(train_data)
    print(df_train["player_won"].value_counts())


except FileNotFoundError:
    print(f"ERROR: Could not find the training file at '{config.DATA_TRAIN_PATH}'.")
    print("Please make sure you have added the competition data to this notebook.")