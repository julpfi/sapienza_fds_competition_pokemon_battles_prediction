import pandas as pd
import numpy as np
# Import the config file (assuming it's in src/utils/config.py)
from src.utils import config 

# Internal Helper Functions 

def _get_type_multiplier(move_type: str, target_types: list) -> float:
    #Calculates the Gen 1 damage multiplier for a move against a target's types
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
    #Calculates the max damage multiplier from a list of attacking types
    if not isinstance(attacking_types, (list, tuple)) or not attacking_types: 
        return 1.0
    if not isinstance(defending_types, (list, tuple)): 
        defending_types = []
        
    multipliers = [_get_type_multiplier(p_type, defending_types) for p_type in attacking_types if p_type] 
    
    return max(multipliers) if multipliers else 1.0

def _get_types_list(row, prefix): 
    """Helper to get a list of types from one-hot encoded columns."""
    return [t.replace(prefix, '').lower() for t in row.index if row[t] == 1]


# Helper 1: k.o. and HP Features

def _create_timeline_features(turns_df: pd.DataFrame) -> pd.DataFrame:
    #Creates aggregated KO and HP % features from the timeline
    
    # KO Count 
    p1_last_status = turns_df.groupby(['battle_id', 'p1_pokemon_state_name'], observed=False)['p1_pokemon_state_status'].last()
    p1_ko_count = p1_last_status[p1_last_status == 'fnt'].groupby('battle_id').count()
    p1_ko_count.name = 'p1_ko_count'
    
    p2_last_status = turns_df.groupby(['battle_id', 'p2_pokemon_state_name'], observed=False)['p2_pokemon_state_status'].last()
    p2_ko_count = p2_last_status[p2_last_status == 'fnt'].groupby('battle_id').count()
    p2_ko_count.name = 'p2_ko_count'
    
    ko_df = pd.merge(p1_ko_count, p2_ko_count, on='battle_id', how='outer')
    ko_df['p1_ko_count'] = ko_df['p1_ko_count'].fillna(0).astype(int)
    ko_df['p2_ko_count'] = ko_df['p2_ko_count'].fillna(0).astype(int)
    ko_df['ko_advantage'] = ko_df['p2_ko_count'] - ko_df['p1_ko_count']
    
    # HP % calc
    p1_last_hp_per_pokemon = turns_df.groupby(['battle_id', 'p1_pokemon_state_name'], observed=False)['p1_pokemon_state_hp_pct'].last()
    p1_team_avg_hp = p1_last_hp_per_pokemon.groupby('battle_id').mean()
    p1_team_avg_hp.name = 'p1_team_avg_hp'
    
    p2_last_hp_per_pokemon = turns_df.groupby(['battle_id', 'p2_pokemon_state_name'], observed=False)['p2_pokemon_state_hp_pct'].last()
    p2_team_avg_hp = p2_last_hp_per_pokemon.groupby('battle_id').mean()
    p2_team_avg_hp.name = 'p2_team_avg_hp'
    
    hp_df = pd.merge(p1_team_avg_hp, p2_team_avg_hp, on='battle_id', how='outer')
    hp_df = hp_df.fillna(0.5) 
    hp_df['team_hp_advantage'] = hp_df['p1_team_avg_hp'] - hp_df['p2_team_avg_hp']
    
    timeline_features_df = pd.merge(ko_df, hp_df, on='battle_id', how='outer')
    return timeline_features_df


# Helper 2: Boost Utility Features

def _create_utility_boost_features(turns_df: pd.DataFrame) -> pd.DataFrame:
    """Creates cumulative features based on *useful* stat boosts."""
    temp_df = turns_df.copy()
    
    # p1 Scores (per turn) 
    p1_atk_boost_used = np.where(temp_df['p1_move_details_category'] == 'physical', temp_df['p1_pokemon_state_boost_atk'], 0)
    p1_spa_boost_used = np.where(temp_df['p1_move_details_category'] == 'special', temp_df['p1_pokemon_state_boost_spa'], 0)
    temp_df['p1_boost_off_turn'] = p1_atk_boost_used + p1_spa_boost_used
    
    p1_def_boost_used = np.where(temp_df['p2_move_details_category'] == 'physical', temp_df['p1_pokemon_state_boost_def'], 0)
    p1_spd_boost_used = np.where(temp_df['p2_move_details_category'] == 'special', temp_df['p1_pokemon_state_boost_spd'], 0)
    temp_df['p1_boost_def_turn'] = p1_def_boost_used + p1_spd_boost_used
    
    temp_df['p1_boost_spe_turn'] = temp_df['p1_pokemon_state_boost_spe']
    
    # p2 Scores (per turn) 
    p2_atk_boost_used = np.where(temp_df['p2_move_details_category'] == 'physical', temp_df['p2_pokemon_state_boost_atk'], 0)
    p2_spa_boost_used = np.where(temp_df['p2_move_details_category'] == 'special', temp_df['p2_pokemon_state_boost_spa'], 0)
    temp_df['p2_boost_off_turn'] = p2_atk_boost_used + p2_spa_boost_used
    
    p2_def_boost_used = np.where(temp_df['p1_move_details_category'] == 'physical', temp_df['p2_pokemon_state_boost_def'], 0)
    p2_spd_boost_used = np.where(temp_df['p1_move_details_category'] == 'special', temp_df['p2_pokemon_state_boost_spd'], 0)
    temp_df['p2_boost_def_turn'] = p2_def_boost_used + p2_spd_boost_used
    
    temp_df['p2_boost_spe_turn'] = temp_df['p2_pokemon_state_boost_spe']
    
    # aggregation (Sum for the whole battle) 
    cols_to_sum = ['p1_boost_off_turn', 'p1_boost_def_turn', 'p1_boost_spe_turn', 
                   'p2_boost_off_turn', 'p2_boost_def_turn', 'p2_boost_spe_turn']
    cumulative_boosts_df = temp_df.groupby('battle_id')[cols_to_sum].sum()
    
    #  Rename 
    cumulative_boosts_df = cumulative_boosts_df.rename(columns={
        'p1_boost_off_turn': 'p1_utility_boost_off', 'p1_boost_def_turn': 'p1_utility_boost_def', 'p1_boost_spe_turn': 'p1_utility_boost_spe',
        'p2_boost_off_turn': 'p2_utility_boost_off', 'p2_boost_def_turn': 'p2_utility_boost_def', 'p2_boost_spe_turn': 'p2_utility_boost_spe'
    })
    
    # differences
    cumulative_boosts_df['utility_boost_off_adv'] = (cumulative_boosts_df['p1_utility_boost_off'] - cumulative_boosts_df['p2_utility_boost_off'])
    cumulative_boosts_df['utility_boost_def_adv'] = (cumulative_boosts_df['p1_utility_boost_def'] - cumulative_boosts_df['p2_utility_boost_def'])
    cumulative_boosts_df['utility_boost_spe_adv'] = (cumulative_boosts_df['p1_utility_boost_spe'] - cumulative_boosts_df['p2_utility_boost_spe'])
    
    return cumulative_boosts_df


# Helper 3: Status Pressure Features

def _create_status_pressure_features(turns_df: pd.DataFrame) -> pd.DataFrame:
    # Calculates the total number of turns each player spent with a Pokemon afflicted by a major status condition.
    
    MAJOR_STATUS = ['par', 'slp', 'frz', 'brn', 'psn', 'tox']
    
    p1_status_turns = turns_df[turns_df['p1_pokemon_state_status'].isin(MAJOR_STATUS)]
    p1_status_turns_total = p1_status_turns.groupby('battle_id').size()
    p1_status_turns_total.name = 'p1_status_turns_total'
    
    p2_status_turns = turns_df[turns_df['p2_pokemon_state_status'].isin(MAJOR_STATUS)]
    p2_status_turns_total = p2_status_turns.groupby('battle_id').size()
    p2_status_turns_total.name = 'p2_status_turns_total'
    
    status_df = pd.merge(p1_status_turns_total, p2_status_turns_total, on='battle_id', how='outer')
    status_df = status_df.fillna(0).astype(int) 
    
    status_df['status_turn_advantage'] = status_df['p2_status_turns_total'] - status_df['p1_status_turns_total']
    
    return status_df


# Helper 4: Dynamic Matchup Features

def _create_dynamic_matchup_features(turns_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    # Calculates P1's STAB count and P2's attack effectiveness against P1's active Pokemon.
    
    temp_df = turns_df.copy()
    
    p1_type_cols = [f'type_{t}' for t in config.POKEMON_TYPES]
    p1_types_df = teams_df[['battle_id', 'name'] + p1_type_cols].copy()
    p1_types_df['p1_types'] = p1_types_df[p1_type_cols].apply(
        lambda row: _get_types_list(row, 'type_'),
        axis=1
    )
    p1_types_map = p1_types_df.groupby('battle_id').apply(
        lambda x: pd.Series(x['p1_types'].values, index=x['name']).to_dict(), include_groups=False
    ).to_dict()

    temp_df['p1_current_types'] = temp_df.apply(
        lambda row: p1_types_map.get(row['battle_id'], {}).get(row['p1_pokemon_state_name'], []),
        axis=1
    )
    
    temp_df['p1_move_type_clean'] = temp_df['p1_move_details_type'].astype(str).str.lower().replace('nan', '')
    temp_df['p2_move_type_clean'] = temp_df['p2_move_details_type'].astype(str).str.lower().replace('nan', '')

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


# Main feature_engineering function (Entry point for main.py)

def feature_engineering_version_3(
    train: bool,
    battles_df: pd.DataFrame, 
    turns_df: pd.DataFrame, 
    teams_df: pd.DataFrame
) -> pd.DataFrame:
    
    # Combines DataFrames, calls all helpers to create features (static + dynamic), and produces the final model-ready DataFrame.
    
    print(" Running feature engineering version 3\n ")
    # Use the passed DataFrame names
    final_df = battles_df.copy()

    #  1. p1 Team Aggregation (Static) 
    print("Aggregating p1 team stats...")
    team_stats_cols = [f"base_{s}" for s in ["hp", "atk", "def", "spa", "spd", "spe"]]
    team_features = teams_df.groupby('battle_id')[team_stats_cols].mean().reset_index()
    team_features.columns = ['battle_id'] + [f'p1_team_mean_{col.replace("base_", "")}' for col in team_stats_cols]
    team_features['p1_mean_offense'] = team_features['p1_team_mean_atk'] + team_features['p1_team_mean_spa']
    final_df = pd.merge(final_df, team_features, on='battle_id', how='left')

    #  2. Turn 1 Aggregation (Static) 
    print("Aggregating Turn 1 features...")
    turn_1_df = turns_df[turns_df['turn'] == 1].copy()
    turn_1_df['p2_statused_turn_1'] = turn_1_df['p2_pokemon_state_status'].apply(lambda x: 1 if pd.notna(x) and x != 'nostatus' else 0)
    turn_1_df['p1_switched_turn_1'] = (turn_1_df['p1_move_details_name'] == 'MISSING_MOVE').astype(int)
    dynamic_features = turn_1_df[['battle_id', 'p2_statused_turn_1', 'p1_switched_turn_1']]
    final_df = pd.merge(final_df, dynamic_features, on='battle_id', how='left')

    #  3. Integrate New Dynamic Features 
    print("Creating K.O. and HP procent features...")
    timeline_features = _create_timeline_features(turns_df)
    final_df = pd.merge(final_df, timeline_features, on='battle_id', how='left')

    print("Creating Utility Boost features...")
    boost_features = _create_utility_boost_features(turns_df)
    final_df = pd.merge(final_df, boost_features, on='battle_id', how='left')
    
    print("Creating Status Pressure features...")
    status_features = _create_status_pressure_features(turns_df)
    final_df = pd.merge(final_df, status_features, on='battle_id', how='left')
    
    print("Creating Dynamic Matchup features...")
    dynamic_matchup_features = _create_dynamic_matchup_features(turns_df, teams_df)
    final_df = pd.merge(final_df, dynamic_matchup_features, on='battle_id', how='left')
    
    #  4. Static differences  
    print("Creating static lead matchup features...")
    final_df['p2_lead_defense'] = final_df['p2_lead_base_def'] + final_df['p2_lead_base_spa']
    final_df['lead_spe_advantage'] = final_df['p1_team_mean_spe'] - final_df['p2_lead_base_spe']
    final_df['p1_off_vs_p2_def_ratio'] = final_df['p1_mean_offense'] / final_df['p2_lead_defense'].replace(0, 1)

    #  5. Static Type Matchup Score  
    p1_lead_df = teams_df[teams_df['pokemon_nr'] == 0].set_index('battle_id')
    p1_types_cols = [f'type_{t}' for t in config.POKEMON_TYPES]
    p2_types_cols = [f'p2_lead_type_{t}' for t in config.POKEMON_TYPES]
    
    p1_lead_types_series = p1_lead_df[p1_types_cols].apply(lambda row: _get_types_list(row, 'type_'), axis=1)
    p2_lead_types_series = final_df[p2_types_cols].apply(lambda row: _get_types_list(row, 'p2_lead_type_'), axis=1) 
    
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

    #  6. Final cleanup
    print("Finalizing feature set and cleaning up...")
    
    # List of all component features that are now redundant
    redundant_component_features = [
        'ko_advantage',
        'p1_team_avg_hp', 'p2_team_avg_hp',
        'p1_utility_boost_off', 'p2_utility_boost_off',
        'p1_utility_boost_def', 'p2_utility_boost_def',
        'p1_utility_boost_spe', 'p2_utility_boost_spe',
        'status_turn_advantage',
        'p1_type_matchup_score', 'p2_type_matchup_score',
        'p1_team_mean_hp', 'p1_team_mean_atk', 'p1_team_mean_def', 
        'p1_team_mean_spa', 'p1_team_mean_spd', 'p1_team_mean_spe',
        'p1_mean_offense',
        'p2_lead_base_hp', 'p2_lead_base_atk', 'p2_lead_base_def',
        'p2_lead_base_spa', 'p2_lead_base_spd', 'p2_lead_base_spe',
        'p2_lead_defense',
        'p2_lead_name', 
        'p2_lead_level', 
    ]
    
    # Drop temp columns, original P2 types, and redundant components
    cols_to_drop = ['p1_lead_types', 'p2_lead_types'] + p2_types_cols + redundant_component_features
    
    final_df = final_df.drop(columns=cols_to_drop, errors='ignore').fillna(0)

    print(f"Feature engineering version 3 complete. Feature count: {final_df.shape[1]}\n")
    return final_df