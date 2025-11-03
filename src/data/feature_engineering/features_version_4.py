import pandas as pd
import numpy as np
from utils.config import POKEMON_TYPES

# 0. Type Calc

def _get_type_multiplier(move_type: str, target_types: list) -> float:
    #Calculates the damage multiplier for a move against a target's types 
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
    
    if not isinstance(target_types, (list, tuple)): 
        target_types = []
    if not isinstance(move_type, str):
        return 1.0 

    move_type_upper = move_type.upper()
    
    for target_type in target_types:
        if not isinstance(target_type, str):
            continue 
            
        target_type_upper = target_type.upper()
        
        if target_type_upper in ['NOTYPE', 'NONE', 'UNKNOWN', 'notype']: 
            continue
            
        if move_type_upper in type_chart: 
            multiplier *= type_chart.get(move_type_upper, {}).get(target_type_upper, 1.0)
        else: 
            multiplier *= 1.0 
            
    return multiplier

def _calculate_matchup_score_robust(attacking_types: list, defending_types: list) -> float:
    # Calculates the max damage multiplier from a list of attacking types
    if not isinstance(attacking_types, (list, tuple)) or not attacking_types: 
        return 1.0
    if not isinstance(defending_types, (list, tuple)): 
        defending_types = []
        
    multipliers = [_get_type_multiplier(p_type, defending_types) for p_type in attacking_types if p_type] 
    
    return max(multipliers) if multipliers else 1.0

def _get_types_list(row, prefix): 
    #  to get a list of types from one-hot encoded columns
    # We need to access the config object defined at the top of the file
    return [t.replace(prefix, '').lower() for t in POKEMON_TYPES if f"{prefix}{t}" in row and row[f"{prefix}{t}"] == 1]


# 1: K.O. and HP %  like v3

def _create_timeline_features(turns_df: pd.DataFrame) -> pd.DataFrame:
    #Creates aggregated KO and HP % features from the timeline
    
    #  KO Count 
    p1_last_status = turns_df.groupby(['battle_id', 'p1_pokemon_state_name'], observed=False)['p1_pokemon_state_status'].last()
    p1_ko_count = p1_last_status[p1_last_status == 'fnt'].groupby('battle_id').count()
    p1_ko_count.name = 'p1_ko_count'
    
    p2_last_status = turns_df.groupby(['battle_id', 'p2_pokemon_state_name'], observed=False)['p2_pokemon_state_status'].last()
    p2_ko_count = p2_last_status[p2_last_status == 'fnt'].groupby('battle_id').count()
    p2_ko_count.name = 'p2_ko_count'
    
    ko_df = pd.merge(p1_ko_count, p2_ko_count, on='battle_id', how='outer')
    ko_df['p1_ko_count'] = ko_df['p1_ko_count'].fillna(0).astype(int)
    ko_df['p2_ko_count'] = ko_df['p2_ko_count'].fillna(0).astype(int)
    
    #  HP % calc 
    p1_last_hp_per_pokemon = turns_df.groupby(['battle_id', 'p1_pokemon_state_name'], observed=False)['p1_pokemon_state_hp_pct'].last()
    p1_team_avg_hp = p1_last_hp_per_pokemon.groupby('battle_id').mean()
    p1_team_avg_hp.name = 'p1_team_avg_hp'
    
    p2_last_hp_per_pokemon = turns_df.groupby(['battle_id', 'p2_pokemon_state_name'], observed=False)['p2_pokemon_state_hp_pct'].last()
    p2_team_avg_hp = p2_last_hp_per_pokemon.groupby('battle_id').mean()
    p2_team_avg_hp.name = 'p2_team_avg_hp'
    
    hp_df = pd.merge(p1_team_avg_hp, p2_team_avg_hp, on='battle_id', how='outer')
    hp_df = hp_df.fillna(0.5) 
    hp_df['team_hp_advantage'] = hp_df['p1_team_avg_hp'] - hp_df['p2_team_avg_hp']
    
    # Only merges the features taht are kept
    timeline_features_df = pd.merge(ko_df, hp_df, on='battle_id', how='outer')
    timeline_features_df = timeline_features_df.drop(columns=['p1_team_avg_hp', 'p2_team_avg_hp'])
    
    return timeline_features_df


# 2: STATUS PRESSURE 

def _create_status_pressure_features_v4(turns_df: pd.DataFrame) -> pd.DataFrame:
    # Calculates total turns spent afflicted by status -> differnetiating "Major" (damage/annoyance) from "Critical" (turn loss

    # "Minor" status (annoying, but you can still move)
    MAJOR_STATUS = ['par', 'brn', 'psn', 'tox']
    # "Critical" status (move-ending, lose your turn)
    CRITICAL_STATUS = ['slp', 'frz']
    
    #  1. Count major status turns 
    p1_major_status_turns = turns_df[turns_df['p1_pokemon_state_status'].isin(MAJOR_STATUS)].groupby('battle_id').size()
    p1_major_status_turns.name = 'p1_major_status_turns'
    p2_major_status_turns = turns_df[turns_df['p2_pokemon_state_status'].isin(MAJOR_STATUS)].groupby('battle_id').size()
    p2_major_status_turns.name = 'p2_major_status_turns'
    
    #  2. Count ciritcal status turns 
    p1_critical_status_turns = turns_df[turns_df['p1_pokemon_state_status'].isin(CRITICAL_STATUS)].groupby('battle_id').size()
    p1_critical_status_turns.name = 'p1_critical_status_turns'
    p2_critical_status_turns = turns_df[turns_df['p2_pokemon_state_status'].isin(CRITICAL_STATUS)].groupby('battle_id').size()
    p2_critical_status_turns.name = 'p2_critical_status_turns'

    # 3. Merge all counts
    status_df = pd.merge(p1_major_status_turns, p2_major_status_turns, on='battle_id', how='outer')
    status_df = pd.merge(status_df, p1_critical_status_turns, on='battle_id', how='outer')
    status_df = pd.merge(status_df, p2_critical_status_turns, on='battle_id', how='outer')
    
    status_df = status_df.fillna(0).astype(int) 
    
    return status_df


#  3: DYNAMIC MATCHUP 

def _create_dynamic_matchup_features(turns_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    # Calculates P1's STAB count and P2's attack effectiveness against P1's active Pokemon.
    
    temp_df = turns_df.copy()
    
    p1_type_cols = [f'type_{t}' for t in POKEMON_TYPES]
    p1_types_df = teams_df[['battle_id', 'name'] + p1_type_cols].copy()
    p1_types_df['p1_types'] = p1_types_df[p1_type_cols].apply(
        lambda row: _get_types_list(row, 'type_'),
        axis=1
    )
    p1_types_map = p1_types_df.groupby('battle_id').apply(
        lambda x: pd.Series(x['p1_types'].values, index=x['name']).to_dict(),
        include_groups=False
    ).to_dict()

    temp_df['p1_current_types'] = temp_df.apply(
        lambda row: p1_types_map.get(row['battle_id'], {}).get(row['p1_pokemon_state_name'], []),
        axis=1
    )
    
    temp_df['p1_move_type_clean'] = temp_df['p1_move_details_type'].astype('object').fillna('').astype(str).str.lower()
    temp_df['p2_move_type_clean'] = temp_df['p2_move_details_type'].astype('object').fillna('').astype(str).str.lower()

    temp_df['p1_stab_flag'] = temp_df.apply(
        lambda row: 1 if (row['p1_move_type_clean'] in row['p1_current_types']) and (row['p1_move_details_category'] not in ['status', 'MISSING_MOVE']) else 0,
        axis=1
    )
    
    temp_df['p2_eff_score'] = temp_df.apply(
        lambda row: _get_type_multiplier(row['p2_move_type_clean'], row['p1_current_types']) if (row['p2_move_details_category'] not in ['status', 'MISSING_MOVE'] and row['p1_current_types']) else 1.0,
        axis=1
    )
    
    agg_dict = {
        'p1_stab_flag': 'sum',
        'p2_eff_score': [
            lambda x: (x > 1).sum(), 
            lambda x: (x < 1).sum()
        ]
    }
    dynamic_matchup_df = temp_df.groupby('battle_id').agg(agg_dict)
    
    dynamic_matchup_df.columns = [
        'p1_stab_count',
        'p2_hits_p1_super_effective', 
        'p2_hits_p1_not_effective'
        ]
    
    return dynamic_matchup_df


# feature_engineering_version_4

def feature_engineering_version_4(
    train: bool,
    battles_df: pd.DataFrame, 
    turns_df: pd.DataFrame, 
    teams_df: pd.DataFrame
) -> pd.DataFrame:
    
    # Creates the final feature set, removing none importnat features (boosts, turn_1_switch) and refining status feature
    
    final_df = battles_df.copy()

    #  1. p1 Team Aggregation (Static) 
    print("Aggregating P1 team stats...")
    team_stats_cols = [f"base_{s}" for s in ["hp", "atk", "def", "spa", "spd", "spe"]]
    team_features = teams_df.groupby('battle_id')[team_stats_cols].mean().reset_index()
    team_features.columns = ['battle_id'] + [f'p1_team_mean_{col.replace("base_", "")}' for col in team_stats_cols]
    team_features['p1_mean_offense'] = team_features['p1_team_mean_atk'] + team_features['p1_team_mean_spa']
    final_df = pd.merge(final_df, team_features, on='battle_id', how='left')

    #  2. Integrate Dynamic Features 
    print("Creating K.O. and HP % features...")
    timeline_features = _create_timeline_features(turns_df)
    final_df = pd.merge(final_df, timeline_features, on='battle_id', how='left')
    
    # 3. Status Pressure 
    print("Creating refined status pressure features...")
    status_features = _create_status_pressure_features_v4(turns_df)
    final_df = pd.merge(final_df, status_features, on='battle_id', how='left')
    
    print("Creating Dynamic Matchup features...")
    dynamic_matchup_features = _create_dynamic_matchup_features(turns_df, teams_df)
    final_df = pd.merge(final_df, dynamic_matchup_features, on='battle_id', how='left')
    
    #  4. Static Differentials  
    print("Creating static lead matchup features...")
    final_df['p2_lead_defense'] = final_df['p2_lead_base_def'] + final_df['p2_lead_base_spa']
    final_df['lead_spe_advantage'] = final_df['p1_team_mean_spe'] - final_df['p2_lead_base_spe']
    final_df['p1_off_vs_p2_def_ratio'] = final_df['p1_mean_offense'] / final_df['p2_lead_defense'].replace(0, 1)

    #  5. Static Type Matchup Score  
    p1_lead_df = teams_df[teams_df['pokemon_nr'] == 0].set_index('battle_id')
    p1_types_cols = [f'type_{t}' for t in POKEMON_TYPES]
    p2_types_cols = [f'p2_lead_type_{t}' for t in POKEMON_TYPES]
    
    p1_lead_types_series = p1_lead_df[p1_types_cols].apply(lambda row: _get_types_list(row, 'type_'), axis=1)
    p2_lead_types_series = final_df[p2_types_cols].apply(lambda row: _get_types_list(row, 'type_'), axis=1) 
    
    final_df['p1_lead_types'] = final_df['battle_id'].map(p1_lead_types_series.to_dict()).fillna('').apply(list)
    final_df['p2_lead_types'] = final_df['battle_id'].map(p2_lead_types_series.to_dict()).fillna('').apply(list)
    
    final_df['p1_type_matchup_score'] = final_df.apply(
        lambda row: _calculate_matchup_score_robust(row['p1_lead_types'], row['p2_lead_types']), 
        axis=1
    )
    final_df['p2_type_matchup_score'] = final_df.apply(
        lambda row: _calculate_matchup_score_robust(row['p2_lead_types'], row['p1_lead_types']), 
        axis=1
    )
    final_df['type_matchup_diff'] = final_df['p1_type_matchup_score'] - final_df['p2_type_matchup_score']
    
    #  6. Static interaction feature 
    print("Creating interaction feature (Speed x Type)...")
    final_df['speed_x_type_adv'] = final_df['lead_spe_advantage'] * final_df['type_matchup_diff']


    redundant_component_features = [
        # Type components
        'p1_type_matchup_score', 'p2_type_matchup_score',
        
        # p1 Stat Components
        'p1_team_mean_hp', 'p1_team_mean_atk', 'p1_team_mean_def', 
        'p1_team_mean_spa', 'p1_team_mean_spd', 'p1_team_mean_spe',
        'p1_mean_offense',

        # p2 Lead Stat Components
        'p2_lead_base_hp', 'p2_lead_base_atk', 'p2_lead_base_def',
        'p2_lead_base_spa', 'p2_lead_base_spd', 'p2_lead_base_spe',
        'p2_lead_defense',
        'p2_lead_name', 
        'p2_lead_level', 
    ]
    
    # Define columns to drop: temp, original P2 types, and redundant components
    cols_to_drop = ['p1_lead_types', 'p2_lead_types'] + p2_types_cols + redundant_component_features
    
    final_df = final_df.drop(columns=cols_to_drop, errors='ignore').fillna(0)
    final_df = final_df.infer_objects(copy=False)

    print(f"Feature enginering version 4 completeed. Feature count: {final_df.shape[1]}")
    return final_df