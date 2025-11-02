from typing import List
import pandas as pd 
from src.utils import config



# --- HELPER FUNCTION 1: GEN 1 TYPE MULTIPLIER ---
def get_type_multiplier(move_type: str, target_types: list) -> float:
    """Calculates the combined damage multiplier (Gen 1 rules) against the target's types."""
    
    # Gen 1 Type Effectiveness Chart (Attacker: {Defender: Multiplier})
    type_chart = {
        'NORMAL': {'ROCK': 0.5, 'GHOST': 0}, 'FIRE': {'GRASS': 2.0, 'ICE': 2.0, 'BUG': 2.0, 'WATER': 0.5, 'ROCK': 0.5, 'DRAGON': 0.5},
        'WATER': {'FIRE': 2.0, 'GROUND': 2.0, 'ROCK': 2.0, 'WATER': 0.5, 'GRASS': 0.5, 'DRAGON': 0.5}, 'GRASS': {'WATER': 2.0, 'GROUND': 2.0, 'ROCK': 2.0, 'FIRE': 0.5, 'GRASS': 0.5, 'POISON': 0.5, 'FLYING': 0.5, 'BUG': 0.5, 'DRAGON': 0.5},
        'ELECTRIC': {'WATER': 2.0, 'FLYING': 2.0, 'GRASS': 0.5, 'ELECTRIC': 0.5, 'DRAGON': 0.5, 'GROUND': 0}, 'ICE': {'GRASS': 2.0, 'GROUND': 2.0, 'FLYING': 2.0, 'DRAGON': 2.0, 'FIRE': 0.5, 'WATER': 0.5, 'ICE': 0.5},
        'FIGHTING': {'NORMAL': 2.0, 'ROCK': 2.0, 'ICE': 2.0, 'FLYING': 0.5, 'POISON': 0.5, 'BUG': 0.5, 'PSYCHIC': 0.5, 'GHOST': 0}, 'POISON': {'GRASS': 2.0, 'FIGHTING': 2.0, 'POISON': 0.5, 'GROUND': 0.5, 'ROCK': 0.5, 'GHOST': 0.5},
        'GROUND': {'FIRE': 2.0, 'ELECTRIC': 2.0, 'POISON': 2.0, 'ROCK': 2.0, 'GRASS': 0.5, 'BUG': 0.5, 'FLYING': 0}, 'FLYING': {'GRASS': 2.0, 'FIGHTING': 2.0, 'BUG': 2.0, 'ELECTRIC': 0.5, 'ROCK': 0.5},
        'PSYCHIC': {'FIGHTING': 2.0, 'POISON': 2.0, 'PSYCHIC': 0.5}, 'BUG': {'GRASS': 2.0, 'PSYCHIC': 2.0, 'FIRE': 0.5, 'FIGHTING': 0.5, 'FLYING': 0.5, 'GHOST': 0.5},
        'ROCK': {'FIRE': 2.0, 'ICE': 2.0, 'FLYING': 2.0, 'BUG': 2.0, 'FIGHTING': 0.5, 'GROUND': 0.5}, 'GHOST': {'GHOST': 2.0, 'NORMAL': 0, 'PSYCHIC': 0}, 
        'DRAGON': {'DRAGON': 2.0},
    }
    multiplier = 1.0
    
    # Robustness check: ensures target_types is iterable (not a float/None)
    if not isinstance(target_types, (list, tuple)):
        target_types = []
        
    for target_type in target_types:
        target_type_upper = target_type.upper() 
        move_type_upper = move_type.upper()
        # Ignore custom types (e.g., 'notype')
        if target_type_upper in ['NOTYPE', 'NONE', 'UNKNOWN', 'notype']: continue
        multiplier *= type_chart.get(move_type_upper, {}).get(target_type_upper, 1.0)
    return multiplier


# --- HELPER FUNCTION 2: ROBUST MATCHUP SCORE CALCULATION ---
def calculate_matchup_score_robust(attacking_types: list, defending_types: list) -> float:
    """
    Calculates the maximum damage multiplier for a set of attacking types 
    against defending types, including robustness checks against non-list inputs.
    """
    # CRITICAL CHECK: Ensures the attacking list is iterable before passing to max()
    if not isinstance(attacking_types, (list, tuple)) or not attacking_types:
        return 1.0 # Neutral multiplier if the attacking list is invalid/empty
    
    if not isinstance(defending_types, (list, tuple)):
         defending_types = []

    # Calculate the maximum multiplier
    return max(get_type_multiplier(p_type, defending_types) for p_type in attacking_types)


# --- MAIN FEATURE ENGINEERING FUNCTION (VERSION 2) ---
def feature_engineering_version_2(
    train:bool,
    battles_df: pd.DataFrame, 
    turns_df: pd.DataFrame, 
    teams_df: pd.DataFrame
) -> pd.DataFrame:
    
    """
    Creates the final feature DataFrame by merging and aggregating the three modular DataFrames.
    This version incorporates robust type handling and advanced Gen 1 logic.
    """
    
    final_df = battles_df.copy()
    
    # 1. P1 Team Aggregation (Mean Stats)
    team_stats_cols = [f"base_{s}" for s in ["hp", "atk", "def", "spa", "spd", "spe"]]
    team_features = teams_df.groupby('battle_id')[team_stats_cols].mean().reset_index()
    team_features.columns = ['battle_id'] + [f'p1_team_mean_{col.replace("base_", "")}' for col in team_stats_cols]
    
    # P1 Mean Offense (Gen 1: Atk + Sp.Atk)
    team_features['p1_mean_offense'] = team_features['p1_team_mean_atk'] + team_features['p1_team_mean_spa']
    final_df = pd.merge(final_df, team_features, on='battle_id', how='left')

    # 2. Turn 1 Dynamic Features (Initial advantage signals)
    turn_1_df = turns_df[turns_df['turn'] == 1].copy()
    turn_1_df['p2_statused_turn_1'] = turn_1_df['p2_pokemon_state_status'].apply(lambda x: 1 if pd.notna(x) and x != 'nostatus' else 0)
    turn_1_df['p1_switched_turn_1'] = turn_1_df['p1_move_details_name'].isna().astype(int)
    dynamic_features = turn_1_df[['battle_id', 'p2_statused_turn_1', 'p1_switched_turn_1']]
    final_df = pd.merge(final_df, dynamic_features, on='battle_id', how='left')

    # 3. Differential and Matchup Calculations
    final_df['p2_lead_defense'] = final_df['p2_lead_base_def'] + final_df['p2_lead_base_spa'] # P2 Lead Defense (Gen 1)
    final_df['lead_spe_advantage'] = final_df['p1_team_mean_spe'] - final_df['p2_lead_base_spe'] # Speed Difference
    final_df['p1_off_vs_p2_def_ratio'] = final_df['p1_mean_offense'] / final_df['p2_lead_defense'].replace(0, 1) # Offense Ratio
    
    # Prepare Type Lists for Matchup Calculation
    p1_lead_df = teams_df[teams_df['pokemon_nr'] == 0].set_index('battle_id')
    p1_types_cols = [f'type_{t}' for t in config.POKEMON_TYPES]
    p2_types_cols = [f'p2_lead_type_{t}' for t in config.POKEMON_TYPES]

    def get_types_list(row, prefix):
        return [t.replace(prefix, '').lower() for t in row.index if row[t] == 1]

    p1_lead_types_series = p1_lead_df[p1_types_cols].apply(lambda row: get_types_list(row, 'type_'), axis=1)
    p2_lead_types_series = final_df[p2_types_cols].apply(lambda row: get_types_list(row, 'p2_lead_type_'), axis=1)

    # Map Type Lists to Final DF (Robustness: fillna('').apply(list) handles missing/NaN types)
    final_df['p1_lead_types'] = final_df['battle_id'].map(p1_lead_types_series.to_dict()).fillna('').apply(list)
    final_df['p2_lead_types'] = final_df['battle_id'].map(p2_lead_types_series.to_dict()).fillna('').apply(list)
    
    # Calculate Type Score using the robust helper function
    final_df['p1_type_matchup_score'] = final_df.apply(
        lambda row: calculate_matchup_score_robust(row['p1_lead_types'], row['p2_lead_types']), 
        axis=1
    )
    final_df['p2_type_matchup_score'] = final_df.apply(
        lambda row: calculate_matchup_score_robust(row['p2_lead_types'], row['p1_lead_types']), 
        axis=1
    )
    
    final_df['type_matchup_diff'] = final_df['p1_type_matchup_score'] - final_df['p2_type_matchup_score']

    # Final cleanup: drop temporary list columns and fill any final NaNs with 0
    final_df = final_df.drop(columns=['p1_lead_types', 'p2_lead_types'] + p2_types_cols, errors='ignore').fillna(0)
    
    return final_df


#train_df = create_advanced_features_from_dfs(train_battles_df, train_turns_df, train_teams_df)
#test_df = create_advanced_features_from_dfs(test_battles_df, test_turns_df, test_teams_df)



#Basic stats of some feature
#print(train_df[['lead_spe_advantage', 'type_matchup_diff', 'p1_off_vs_p2_def_ratio','p2_statused_turn_1', 'p1_switched_turn_1']].describe())