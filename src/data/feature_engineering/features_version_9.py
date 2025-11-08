import pandas as pd
import numpy as np
from src.utils.config import POKEMON_TYPES 


# ---------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Helper Methods  -----------------------------------------------


def _get_type_multiplier(move_type: str, target_types: list) -> float:
    # Calculates the damage multiplier for a move against a target's types 
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

    move_type = move_type.upper()    
    for target_type in target_types:
        target_type = target_type.upper()
        if not isinstance(target_type, str) or target_type in ['NOTYPE', 'NONE', 'UNKNOWN', 'notype']: 
            continue  
        if move_type in type_chart: 
            multiplier *= type_chart.get(target_type, {}).get(target_type, 1.0)   

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
    # Helper to get a list of types from one-hot encoded columns
    return [t.replace(prefix, '').lower() for t in POKEMON_TYPES if f"{prefix}{t}" in row and row[f"{prefix}{t}"] == 1]


def _build_speed_lookup_map(teams_df: pd.DataFrame, battles_df: pd.DataFrame) -> pd.Series:
    p1_stats = teams_df[['name', 'base_spe']].drop_duplicates(subset=['name'])
    
    p2_stats = battles_df[['p2_lead_name', 'p2_lead_base_spe']].rename(columns={'p2_lead_name': 'name', 'p2_lead_base_spe': 'base_spe'})
    
    lookup_speed_all = pd.concat([p1_stats, p2_stats]).drop_duplicates(subset=['name'])

    map_name2speed = lookup_speed_all.set_index('name')['base_spe']
    return map_name2speed



# ---------------------------------------------------------------------------------------------------------------
# -------------------------------------- Feature Creators ------------------------------------------------------

# 1. P1 Team Stat Features
def _create_team_stat_features(teams_df: pd.DataFrame) -> pd.DataFrame:
    # Creates aggregated stat features for P1's team
    team_stats_cols = [f"base_{s}" for s in ["hp", "atk", "def", "spa", "spd", "spe"]]
    team_features = teams_df.groupby('battle_id')[team_stats_cols].mean().reset_index()
    
    # Rename columns for clarity
    team_features.columns = ['battle_id'] + [f'p1_team_mean_{col.replace("base_", "")}' for col in team_stats_cols]
    
    # Add offensive power metric
    team_features['p1_mean_offense'] = team_features['p1_team_mean_atk'] + team_features['p1_team_mean_spa']
    return team_features



# 2 Timlime Features: KO and HP% 
def _create_timeline_features(turns_df: pd.DataFrame) -> pd.DataFrame:
    # Creates aggregated KO and HP % features from the timeline.
    # Includes both team average HP (relative) and team sum HP (total) as well as ko counts and ko advantage
    
    # KO count 
    p1_last_status = turns_df.groupby(['battle_id', 'p1_pokemon_state_name'], observed=False)['p1_pokemon_state_status'].last()
    p1_ko_count = p1_last_status[p1_last_status == 'fnt'].groupby('battle_id').count()
    p1_ko_count.name = 'p1_ko_count'
    
    p2_last_status = turns_df.groupby(['battle_id', 'p2_pokemon_state_name'], observed=False)['p2_pokemon_state_status'].last()
    p2_ko_count = p2_last_status[p2_last_status == 'fnt'].groupby('battle_id').count()
    p2_ko_count.name = 'p2_ko_count'
    
    ko_df = pd.merge(p1_ko_count, p2_ko_count, on='battle_id', how='outer')
    ko_df['p1_ko_count'] = ko_df['p1_ko_count'].fillna(0).astype(int)
    ko_df['p2_ko_count'] = ko_df['p2_ko_count'].fillna(0).astype(int)

    # Create the "Pokemon Left" features based on the KO counts
    ko_df['p1_pokemon_left'] = 6 - ko_df['p2_ko_count']
    ko_df['p2_pokemon_left'] = 6 - ko_df['p1_ko_count']
    
    # KO Advantage: Positive means P1 has an advantage (P2 has more KOs)
    ko_df['ko_advantage'] = ko_df['p2_ko_count'] - ko_df['p1_ko_count']
    
    # HP %
    # Get the last HP% for every pokemon that participated
    p1_last_hp_per_pokemon = turns_df.groupby(['battle_id', 'p1_pokemon_state_name'], observed=False)['p1_pokemon_state_hp_pct'].last()
    p2_last_hp_per_pokemon = turns_df.groupby(['battle_id', 'p2_pokemon_state_name'], observed=False)['p2_pokemon_state_hp_pct'].last()
    
    # Calculate average hp %
    p1_team_avg_hp = p1_last_hp_per_pokemon.groupby('battle_id').mean()
    p1_team_avg_hp.name = 'p1_team_avg_hp'
    p2_team_avg_hp = p2_last_hp_per_pokemon.groupby('battle_id').mean()
    p2_team_avg_hp.name = 'p2_team_avg_hp'
    
    # Calculate total hp %
    p1_team_sum_hp = p1_last_hp_per_pokemon.groupby('battle_id').sum()
    p1_team_sum_hp.name = 'p1_team_sum_hp'
    p2_team_sum_hp = p2_last_hp_per_pokemon.groupby('battle_id').sum()
    p2_team_sum_hp.name = 'p2_team_sum_hp'

    # Start with avg
    hp_df = pd.merge(p1_team_avg_hp, p2_team_avg_hp, on='battle_id', how='outer')
    # Add sum
    hp_df = pd.merge(hp_df, p1_team_sum_hp, on='battle_id', how='outer')
    hp_df = pd.merge(hp_df, p2_team_sum_hp, on='battle_id', how='outer')
    
    # FillNa: 0.5 for AVG (neutral), 0.0 for SUM (no HP stat)
    hp_df['p1_team_avg_hp'] = hp_df['p1_team_avg_hp'].fillna(0.5)
    hp_df['p2_team_avg_hp'] = hp_df['p2_team_avg_hp'].fillna(0.5)
    hp_df['p1_team_sum_hp'] = hp_df['p1_team_sum_hp'].fillna(0.0)
    hp_df['p2_team_sum_hp'] = hp_df['p2_team_sum_hp'].fillna(0.0)

    # Create advantage features
    hp_df['team_hp_advantage'] = hp_df['p1_team_avg_hp'] - hp_df['p2_team_avg_hp']
    hp_df['team_hp_sum_advantage'] = hp_df['p1_team_sum_hp'] - hp_df['p2_team_sum_hp']
    
    # Merge KO features and HP features
    timeline_features_df = pd.merge(ko_df, hp_df, on='battle_id', how='outer')
    return timeline_features_df



# 3 Satuts Pressure
def _create_status_pressure_features(turns_df: pd.DataFrame) -> pd.DataFrame:
    # Calculates total turns spent afflicted by status, differentiating by severit
    # "Major" status (annoying, but you can still move)
    MAJOR_STATUS = ['par', 'brn', 'psn', 'tox']
    # "Critical" status (move-ending, lose your turn)
    CRITICAL_STATUS = ['slp', 'frz']
    
    # 1. Count "Major" status turns 
    p1_major_status_turns = turns_df[turns_df['p1_pokemon_state_status'].isin(MAJOR_STATUS)].groupby('battle_id').size()
    p1_major_status_turns.name = 'p1_major_status_turns'
    p2_major_status_turns = turns_df[turns_df['p2_pokemon_state_status'].isin(MAJOR_STATUS)].groupby('battle_id').size()
    p2_major_status_turns.name = 'p2_major_status_turns'
    
    # 2. Count "Critical" status turns 
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


# 4 Dynamic Matchup
def _create_dynamic_matchup_features(turns_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    # Calculates P1's STAB count and P2's attack effectiveness against P1's active pokemon
    temp_df = turns_df.copy()
    
    p1_type_cols = [f'type_{t}' for t in POKEMON_TYPES]
    p1_types_df = teams_df[['battle_id', 'name'] + p1_type_cols].copy()
    p1_types_df['p1_types'] = p1_types_df[p1_type_cols].apply(lambda row: _get_types_list(row, 'type_'), axis=1 )
    p1_types_map = p1_types_df.groupby('battle_id').apply(
        lambda x: pd.Series(x['p1_types'].values, index=x['name']).to_dict(), include_groups=False).to_dict()

    temp_df['p1_current_types'] = temp_df.apply(
        lambda row: p1_types_map.get(row['battle_id'], {}).get(row['p1_pokemon_state_name'], []), axis=1)
    
    temp_df['p1_move_type_clean'] = temp_df['p1_move_details_type'].astype('object').fillna('').astype(str).str.lower()
    temp_df['p2_move_type_clean'] = temp_df['p2_move_details_type'].astype('object').fillna('').astype(str).str.lower()

    temp_df['p1_stab_flag'] = temp_df.apply(
        lambda row: 1 
        if (row['p1_move_type_clean'] in row['p1_current_types']) and (row['p1_move_details_category'] not in ['status', 'MISSING_MOVE']) 
        else 0, axis=1)
    
    temp_df['p2_eff_score'] = temp_df.apply(
        lambda row: _get_type_multiplier(row['p2_move_type_clean'], row['p1_current_types']) 
        if (row['p2_move_details_category'] not in ['status', 'MISSING_MOVE'] and row['p1_current_types']) 
        else 1.0, axis=1)
    
    agg_dict = {
        'p1_stab_flag': 'sum',
        'p2_eff_score': [lambda x: (x > 1).sum(), lambda x: (x < 1).sum()]
        }
    
    dynamic_matchup_df = temp_df.groupby('battle_id').agg(agg_dict)
    
    dynamic_matchup_df.columns = [
        'p1_stab_count',
        'p2_hits_p1_super_effective', 
        'p2_hits_p1_not_effective'
        ]
    
    return dynamic_matchup_df


# 5. Lead Differentials Features 
def _create_lead_differential_features(battles_df: pd.DataFrame, team_features_df: pd.DataFrame) -> pd.DataFrame:
    # Creates stat differential features between P1's team and P2's lead
    diff_df = battles_df.copy()
    
    # Defensive total for P2's lead
    diff_df['p2_lead_defense'] = diff_df['p2_lead_base_def'] + diff_df['p2_lead_base_spa']
    
    # Speed advantage (P1 team mean vs P2 lead)
    diff_df['lead_spe_advantage'] = team_features_df['p1_team_mean_spe'] - diff_df['p2_lead_base_spe']
    
    # Offensive power ratio
    diff_df['p1_off_vs_p2_def_ratio'] = team_features_df['p1_mean_offense'] / diff_df['p2_lead_defense'].replace(0, 1)
    
    return diff_df[['battle_id', 'p2_lead_defense', 'lead_spe_advantage', 'p1_off_vs_p2_def_ratio']]


# 6. Type Matchup Features
def _create_lead_type_matchup_features(battles_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    # Creates type effectiveness features between P1 and P2 leads
    matchup_df = battles_df[['battle_id']].copy()
    
    # Get P1's lead types
    p1_lead_df = teams_df[teams_df['pokemon_nr'] == 0].set_index('battle_id')
    p1_types_cols = [f'type_{t}' for t in POKEMON_TYPES]
    p2_types_cols = [f'p2_lead_type_{t}' for t in POKEMON_TYPES]
    
    # Extract type lists
    p1_lead_types_series = p1_lead_df[p1_types_cols].apply(lambda row: _get_types_list(row, 'type_'), axis=1)
    p2_lead_types_series = battles_df[p2_types_cols].apply(lambda row: _get_types_list(row, 'type_'), axis=1)
    
    # Map types to battles
    matchup_df['p1_lead_types'] = matchup_df['battle_id'].map(p1_lead_types_series.to_dict()).fillna('').apply(list)
    matchup_df['p2_lead_types'] = matchup_df['battle_id'].map(p2_lead_types_series.to_dict()).fillna('').apply(list)
    
    # Calculate matchup scores
    matchup_df['p1_type_matchup_score'] = matchup_df.apply(
        lambda row: _calculate_matchup_score_robust(row['p1_lead_types'], row['p2_lead_types']), axis=1)
    matchup_df['p2_type_matchup_score'] = matchup_df.apply(
        lambda row: _calculate_matchup_score_robust(row['p2_lead_types'], row['p1_lead_types']), axis=1)
    
    # Calculate type advantage
    matchup_df['type_matchup_diff'] = matchup_df['p1_type_matchup_score'] - matchup_df['p2_type_matchup_score']
    
    return matchup_df[['battle_id', 'type_matchup_diff']]


# 7. Speed Dynamics Features - First Move Advantage
def _get_first_move_advantage(battles_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    
    speed_df = battles_df[['battle_id', 'p2_lead_base_spe']].copy()
    p1_lead_spe_series = teams_df[teams_df['pokemon_nr'] == 0].set_index('battle_id')['base_spe']
    
    speed_df['p1_lead_base_spe'] = speed_df['battle_id'].map(p1_lead_spe_series)
    speed_df['p1_lead_base_spe'] = speed_df['p1_lead_base_spe'].fillna(0)
    speed_df['p2_lead_base_spe'] = speed_df['p2_lead_base_spe'].fillna(0)

    conditions = [
        speed_df['p1_lead_base_spe'] > speed_df['p2_lead_base_spe'],
        speed_df['p2_lead_base_spe'] > speed_df['p1_lead_base_spe']
        ]

    speed_df['first_move_advantage'] = np.select(conditions, [1, -1], default=0)
    speed_df['lead_spe_diff'] = speed_df['p1_lead_base_spe'] - speed_df['p2_lead_base_spe']
    final_features_df = speed_df.drop(columns=['p1_lead_base_spe', 'p2_lead_base_spe'])
    return final_features_df.set_index('battle_id')


# 8. Dynamic Speed Features - First Move Ratio
def _create_dynamic_speed_features(battles_df, teams_df, turns_df: pd.DataFrame) -> pd.DataFrame:
    # Speed boosts are added via a multiplier 
    # (https://www.pokebeach.com/forums/threads/how-to-calculate-speed.126779/#:~:text=Now%20for%20the%20point%20of,%2D6%20Stage%20=%20*0.25)
    
    map_name2speed = _build_speed_lookup_map(teams_df, battles_df)
    speed_stage_multipliers = {
        -6: 0.25, -5: 0.28, -4: 0.33, -3: 0.4, -2: 0.5, -1: 0.66, 0: 1.0,
         1: 1.5,  2: 2.0,   3: 2.5,  4: 3.0,  5: 3.5,  6: 4.0
    }

    active_turns_df = turns_df[((turns_df['p1_move_details_name'] != 'MISSING_MOVE') | (turns_df['p2_move_details_name'] != 'MISSING_MOVE'))].copy()
    
    active_turns_df['p1_active_spe'] = active_turns_df['p1_pokemon_state_name'].map(map_name2speed).fillna(0)
    active_turns_df['p2_active_spe'] = active_turns_df['p2_pokemon_state_name'].map(map_name2speed).fillna(0)
    
    active_turns_df['p1_priority'] = active_turns_df['p1_move_details_priority'].fillna(0)
    active_turns_df['p2_priority'] = active_turns_df['p2_move_details_priority'].fillna(0)
    
    p1_spe_boost = active_turns_df['p1_pokemon_state_boost_spe'].fillna(0)
    p2_spe_boost = active_turns_df['p2_pokemon_state_boost_spe'].fillna(0)
    
    p1_spe_multiplier = p1_spe_boost.map(speed_stage_multipliers).fillna(1.0)
    p2_spe_multiplier = p2_spe_boost.map(speed_stage_multipliers).fillna(1.0)

    active_turns_df['p1_final_spe'] = active_turns_df['p1_active_spe'] * p1_spe_multiplier
    active_turns_df['p2_final_spe'] = active_turns_df['p2_active_spe'] * p2_spe_multiplier

    conditions = [
        active_turns_df['p1_priority'] > active_turns_df['p2_priority'],
        active_turns_df['p2_priority'] > active_turns_df['p1_priority'],
        active_turns_df['p1_final_spe'] > active_turns_df['p2_final_spe'],
        active_turns_df['p2_final_spe'] > active_turns_df['p1_final_spe']
    ]
    

    active_turns_df['p1_moves_first_flag'] = np.select(conditions, [1, 0, 1, 0], default=0.5)
    
    battle_agg = active_turns_df.groupby('battle_id')['p1_moves_first_flag'].agg(['sum', 'count'])
    battle_agg['dynamic_first_move_ratio'] = battle_agg['sum'] / battle_agg['count']
    
    final_df = battle_agg[['dynamic_first_move_ratio']].fillna(0.5)
    return final_df


# ---------------------------------------------------------------------------------------------------------------
# ---------------------------- Main feature engineering function -----------------------------------------------
def feature_engineering_version_9(
    train: bool,
    battles_df: pd.DataFrame, 
    turns_df: pd.DataFrame, 
    teams_df: pd.DataFrame
) -> pd.DataFrame:
    
    final_df = battles_df.copy()

    # 1. p1 Team Aggregation (Static) 
    print("Aggregating p1 team stats...")
    team_features = _create_team_stat_features(teams_df)
    final_df = pd.merge(final_df, team_features, on='battle_id', how='left')

    # 2. Integrate Dynamic Features 
    print("Creating timeline feature for ko and hp%...")
    timeline_features = _create_timeline_features(turns_df)
    final_df = pd.merge(final_df, timeline_features, on='battle_id', how='left')
    
    # 3. Status Pressure
    print("Creating status pressure features...")
    status_features = _create_status_pressure_features(turns_df)
    final_df = pd.merge(final_df, status_features, on='battle_id', how='left')
    
    # 4. Dynamic Matchup Features
    print("Creating dynamic matchup features...")
    dynamic_matchup_features = _create_dynamic_matchup_features(turns_df, teams_df)
    final_df = pd.merge(final_df, dynamic_matchup_features, on='battle_id', how='left')
    
    # 5. Lead Differentials
    print("Creating lead matchup features...")
    lead_diff_features = _create_lead_differential_features(final_df, team_features)
    final_df = pd.merge(final_df, lead_diff_features, on='battle_id', how='left')
    
    # 6. Type Matchup Features
    print("Creating type matchup features...")
    type_matchup_features = _create_lead_type_matchup_features(final_df, teams_df)
    final_df = pd.merge(final_df, type_matchup_features, on='battle_id', how='left')

    # 7. Speed Dynamics Features - First Move Advantage
    print("Creating speed dynamics features...")
    first_move_advantage_df = _get_first_move_advantage(battles_df, teams_df)
    final_df = final_df.merge(first_move_advantage_df, on='battle_id', how='left')

    # 8. Speed Dynamics Features - First Move Ratio
    dynamic_speed_features_df = _create_dynamic_speed_features(battles_df, teams_df, turns_df)
    final_df = final_df.merge(dynamic_speed_features_df, on='battle_id', how='left')  

    # 9. Static interaction feature 
    print("Creating interaction feature (Speed x Type)...")
    final_df['speed_x_type_adv'] = final_df['lead_spe_advantage'] * final_df['type_matchup_diff']

    # ------------------------- Final Cleanup --------------------------
    lead_types_col = ['p1_lead_types', 'p2_lead_types'] 
    p2_types_cols = [f'p2_lead_type_{t}' for t in POKEMON_TYPES]
    redundant_component_features = [
        # Type components
        'p1_type_matchup_score', 'p2_type_matchup_score',
        
        # p1 Stat Components
        'p1_team_mean_atk', 
        'p1_team_mean_spa', 'p1_team_mean_spe',
        'p1_mean_offense',

        # p2 Lead Stat Components
        'p2_lead_base_def',
        'p2_lead_base_spa', 'p2_lead_base_spe',
        'p2_lead_defense',
        'p2_lead_name',      
    ]
    
    # Define columns to drop: temp, original P2 types, and redundant components
    cols_to_drop = lead_types_col + p2_types_cols + redundant_component_features
    final_df = final_df.drop(columns=cols_to_drop, errors='ignore').fillna(0)

    final_df = final_df.infer_objects(copy=False)
    print(f"Feature engineering version 9 completed. Feature count: {final_df.shape[1]}")
    return final_df