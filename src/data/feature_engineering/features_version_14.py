import pandas as pd
import numpy as np
from src.utils.config import POKEMON_TYPES 


# ---------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Helper Methods  -----------------------------------------------
def _create_pokemon_stats_map(teams_df: pd.DataFrame, battles_df: pd.DataFrame) -> (dict, dict):
    """
    Creates a master dictionary mapping Pokemon names to their types AND base stats.
    Also calculates an 'average' pokemon stat profile for imputation.
    This MUST be built ONLY from training data to prevent data leakage.
    """
    print("Building Pokemon Stats Map (Pokédex)...")
    pokemon_stats_map = {}
    stats_cols = ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']
    
    # 1. Scan the P1 teams_df
    p1_type_cols = [f'type_{t}' for t in POKEMON_TYPES]
    for _, row in teams_df.iterrows():
        name = row['name']
        if name not in pokemon_stats_map:
            stats = {'types': [t.replace('type_', '').lower() for t in POKEMON_TYPES if f"type_{t}" in row and row[f"type_{t}"] == 1]}
            for stat_col in stats_cols:
                stats[stat_col] = row[stat_col]
            pokemon_stats_map[name] = stats
            
    # 2. Scan the P2 battles_df (for the leads)
    p2_type_cols = [f'p2_lead_type_{t}' for t in POKEMON_TYPES]
    for _, row in battles_df.iterrows():
        name = row['p2_lead_name']
        if name not in pokemon_stats_map:
            stats = {'types': [t.replace('p2_lead_type_', '').lower() for t in POKEMON_TYPES if f"p2_lead_type_{t}" in row and row[f"p2_lead_type_{t}"] == 1]}
            for stat_col in stats_cols:
                p2_col_name = f'p2_lead_{stat_col}' # e.g., p2_lead_base_hp
                stats[stat_col] = row[p2_col_name] # e.g., 'base_hp' = row['p2_lead_base_hp']
            pokemon_stats_map[name] = stats
            
    print(f"Pokédex built. Total Pokemon found: {len(pokemon_stats_map)}")
    
    # 3. Create the average/default profile for imputation
    if not pokemon_stats_map:
        print("Warning: Pokédex is empty. Returning default stats.")
        default_pokemon_stats = {col: 0 for col in stats_cols}
        default_pokemon_stats['types'] = []
        return {}, default_pokemon_stats

    temp_df = pd.DataFrame.from_dict(pokemon_stats_map, orient='index')
    avg_stats = temp_df[stats_cols].mean().to_dict()
    default_pokemon_stats = avg_stats
    default_pokemon_stats['types'] = [] # Unknown pokemon has no types for multiplier
    
    print(f"Calculated average stats for imputation.")
    
    return pokemon_stats_map, default_pokemon_stats




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
            multiplier *= type_chart.get(move_type, {}).get(target_type, 1.0)   

    return multiplier

# 1 Timelime Features: KO and HP% 
def _create_timeline_features(turns_df: pd.DataFrame, pokemon_stats_map: dict, default_pokemon_stats: dict) -> pd.DataFrame:
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


    # --- HP % ---
    
    #Dati di base
    base_hp_map = {name: stats['base_hp'] for name, stats in pokemon_stats_map.items()}
    default_hp = default_pokemon_stats.get('base_hp', 100.0) # Fallback

    p1_last_hp_per_pokemon = turns_df.groupby(['battle_id', 'p1_pokemon_state_name'], observed=False)['p1_pokemon_state_hp_pct'].last()
    p2_last_hp_per_pokemon = turns_df.groupby(['battle_id', 'p2_pokemon_state_name'], observed=False)['p2_pokemon_state_hp_pct'].last()
    
    # --- 2b. LOGICA MEDIA PESATA (per 'team_avg_hp' come da tua richiesta precedente) ---
    
    # Pesi P1 (base_hp)
    p1_names = p1_last_hp_per_pokemon.index.get_level_values('p1_pokemon_state_name')
    p1_base_hps_weights = pd.Series(p1_names.map(base_hp_map).fillna(default_hp).values, index=p1_last_hp_per_pokemon.index, name='base_hp')
    
    # Pesi P2 (base_hp)
    p2_names = p2_last_hp_per_pokemon.index.get_level_values('p2_pokemon_state_name')
    p2_base_hps_weights = pd.Series(p2_names.map(base_hp_map).fillna(default_hp).values, index=p2_last_hp_per_pokemon.index, name='base_hp')

    # Calcolo Media Pesata P1: sum(pct * weight) / sum(weight)
    p1_numerator = (p1_last_hp_per_pokemon * p1_base_hps_weights).groupby('battle_id').sum()
    p1_denominator = p1_base_hps_weights.groupby('battle_id').sum()
    p1_team_weighted_avg_hp = (p1_numerator / p1_denominator)
    p1_team_weighted_avg_hp.name = 'p1_team_weighted_avg_hp' 

    # Calcolo Media Pesata P2
    p2_numerator = (p2_last_hp_per_pokemon * p2_base_hps_weights).groupby('battle_id').sum()
    p2_denominator = p2_base_hps_weights.groupby('battle_id').sum()
    p2_team_weighted_avg_hp = (p2_numerator / p2_denominator)
    p2_team_weighted_avg_hp.name = 'p2_team_weighted_avg_hp' 

    # Merge HP features 
    hp_df = pd.merge(p1_team_weighted_avg_hp, p2_team_weighted_avg_hp, on='battle_id', how='outer')
    #hp_df = pd.merge(hp_df, p1_team_sum_hp, on='battle_id', how='outer')
    #hp_df = pd.merge(hp_df, p2_team_sum_hp, on='battle_id', how='outer')
    
    # 5. FillNa 
    hp_df['p1_team_weighted_avg_hp'] = hp_df['p1_team_weighted_avg_hp'].fillna(0.5) # (neutral)
    hp_df['p2_team_weighted_avg_hp'] = hp_df['p2_team_weighted_avg_hp'].fillna(0.5) # (neutral)
    
    # If sum_hp is NaN, it means 0 participants. 
    # The sum for both should be 6.0 (6 Pokemon at 100% HP).
    #hp_df['p1_team_sum_hp'] = hp_df['p1_team_sum_hp'].fillna(6.0) 
    #hp_df['p2_team_sum_hp'] = hp_df['p2_team_sum_hp'].fillna(6.0) 

    # 6. Create advantage features 
    hp_df['team_hp_advantage'] = hp_df['p1_team_weighted_avg_hp'] - hp_df['p2_team_weighted_avg_hp']
    #hp_df['team_hp_sum_advantage'] = hp_df['p1_team_sum_hp'] - hp_df['p2_team_sum_hp']
    
    # 7. Merge KO features and HP features (Unchanged)
    timeline_features_df = pd.merge(ko_df, hp_df, on='battle_id', how='outer')
    return timeline_features_df


#2 Status Features: Status count an advantage
def _create_status_features(turns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the number of UNIQUE POKÉMON afflicted by each status
    during the battle (up to turn 30).
    It also creates "advantage" features for each status.
    """
    
    # Define all status conditions we want to track individually
    STATUS_LIST = ['brn', 'par', 'psn', 'tox', 'slp', 'frz']
    
    all_status_series = [] # List to hold all our partial Series

    # 1. Loop to generate counts for each status
    for status in STATUS_LIST:
        
        # --- Player 1 ---
        p1_col_name = f'p1_{status}_count' # e.g., 'p1_brn_count'
        
        # Key logic: find rows with the status, group by battle,
        # and count the UNIQUE Pokémon names
        p1_series = turns_df[
            turns_df['p1_pokemon_state_status'] == status
        ].groupby('battle_id')['p1_pokemon_state_name'].nunique()
        
        p1_series.name = p1_col_name
        
        # --- Player 2 ---
        p2_col_name = f'p2_{status}_count' # e.g., 'p2_brn_count'
        
        p2_series = turns_df[
            turns_df['p2_pokemon_state_status'] == status
        ].groupby('battle_id')['p2_pokemon_state_name'].nunique()
        
        p2_series.name = p2_col_name
        
        # Add the two series (P1 and P2) to our list
        all_status_series.extend([p1_series, p2_series])

    # 2. Concatenate everything into a single DataFrame
    # pd.concat is very efficient. It joins all Series
    # aligning them by 'battle_id' (which is the index)
    status_df = pd.concat(all_status_series, axis=1)
    
    # 3. Fill zeros and convert to integer
    # If a battle had no 'brn', its count will be 'NaN'.
    # fillna(0) sets it to 0.
    status_df = status_df.fillna(0).astype(int)

    # 4. Create Advantage Features
    #    A positive value = P1 advantage (P2 has more afflicted Pokémon)
    for status in STATUS_LIST:
        p1_col = f'p1_{status}_count'
        p2_col = f'p2_{status}_count'
        adv_col = f'{status}_advantage' # e.g., 'brn_advantage'
        
        status_df[adv_col] = status_df[p2_col] - status_df[p1_col]
        
    print(f"Individual status feature creation complete. {len(status_df.columns)} features created.")
    return status_df


# 3. Switch Pressure Features
def _create_switch_pressure_features(turns_df: pd.DataFrame) -> pd.DataFrame:
    
    #Calculates the total number of *strategic* switches for each player.
    #A strategic switch is a change of Pokemon when the previous one was NOT fainted.
    #This logic is more robust than checking for 'MISSING_MOVE'.
    
    
    # Ensure data is sorted to compare T vs T-1
    df = turns_df.sort_values(by=['battle_id', 'turn']).copy()
    
    # --- 1. Get Previous Turn Info ---
    # We groupby 'battle_id' to prevent .shift() from pulling data from the previous battle
    df['p1_prev_name'] = df.groupby('battle_id')['p1_pokemon_state_name'].shift(1)
    df['p1_prev_status'] = df.groupby('battle_id')['p1_pokemon_state_status'].shift(1)
    
    df['p2_prev_name'] = df.groupby('battle_id')['p2_pokemon_state_name'].shift(1)
    df['p2_prev_status'] = df.groupby('battle_id')['p2_pokemon_state_status'].shift(1)

    # --- 2. Identify Strategic Switches ---
    # A switch is strategic if:
    # 1. The name is different from the previous turn
    # 2. The previous status was NOT 'fnt' (fainted)
    # 3. It's not Turn 1 (where prev_name is NaN)
    
    p1_strategic_switch_bool = (df['p1_pokemon_state_name'] != df['p1_prev_name']) & \
                               (df['p1_prev_status'] != 'fnt') & \
                               (df['turn'] > 1)
    
    p1_switches = p1_strategic_switch_bool.groupby(df['battle_id']).sum()
    p1_switches.name = 'p1_switch_count'
    
    p2_strategic_switch_bool = (df['p2_pokemon_state_name'] != df['p2_prev_name']) & \
                               (df['p2_prev_status'] != 'fnt') & \
                               (df['turn'] > 1)

    p2_switches = p2_strategic_switch_bool.groupby(df['battle_id']).sum()
    p2_switches.name = 'p2_switch_count'
    
    # --- 3. Merge and create the differential ---
    switch_df = pd.merge(p1_switches, p2_switches, on='battle_id', how='outer')
    switch_df = switch_df.fillna(0).astype(int)
    
    # A positive number means P1 switched more (bad)
    # A negative number means P2 switched more (good)
    switch_df['switch_advantage'] = switch_df['p2_switch_count'] - switch_df['p1_switch_count']
    return switch_df


# 4. Last Turn Boost Advantage Features
def _create_last_turn_boost_advantage(turns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the boost advantage based ONLY on the field state
    of the last available turn for each battle.
    """
    
    # 1. Define the stats we are interested in
    boost_stats = ['atk', 'def', 'spa', 'spd', 'spe']
    p1_boost_cols = [f'p1_pokemon_state_boost_{s}' for s in boost_stats]
    p2_boost_cols = [f'p2_pokemon_state_boost_{s}' for s in boost_stats]
    
    # 2. Isolate the LAST turn for each battle
    # This is the most robust way to get the last known state
    # (whether it's turn 30, or earlier if the battle ended)
    last_turn_df = turns_df.loc[turns_df.groupby('battle_id')['turn'].idxmax()]

    # 3. Select only the boost columns and set the index
    # Fill NaNs with 0 (e.g., if a Pokémon is fainted, its boosts are 0)
    boost_df = last_turn_df[['battle_id'] + p1_boost_cols + p2_boost_cols].copy()
    boost_df = boost_df.fillna(0)
    boost_df = boost_df.set_index('battle_id')

    # 4. Create a clean DataFrame for the final features
    final_features_df = pd.DataFrame(index=boost_df.index)

    # 5. Calculate the aggregated advantages (as in your original function)
    
    # Offensive Advantage
    p1_offense_boost = boost_df[f'p1_pokemon_state_boost_atk'] + boost_df[f'p1_pokemon_state_boost_spa']
    p2_offense_boost = boost_df[f'p2_pokemon_state_boost_atk'] + boost_df[f'p2_pokemon_state_boost_spa']
    final_features_df['offense_boost_advantage'] = p1_offense_boost - p2_offense_boost

    # Defensive Advantage
    p1_defense_boost = boost_df[f'p1_pokemon_state_boost_def'] + boost_df[f'p1_pokemon_state_boost_spd']
    p2_defense_boost = boost_df[f'p2_pokemon_state_boost_def'] + boost_df[f'p2_pokemon_state_boost_spd']
    final_features_df['defense_boost_advantage'] = p1_defense_boost - p2_defense_boost
    
    # Speed Advantage
    final_features_df['speed_boost_advantage'] = boost_df[f'p1_pokemon_state_boost_spe'] - boost_df[f'p2_pokemon_state_boost_spe']
    
    # Convert to integer as in your original function
    return final_features_df.astype(int)


# 5. Trapping Features
def _create_trapping_features(turns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts the number of unique Pokemon (per player) that were afflicted
    by a trapping move (clamp, wrap, firespin) at any point.
    
    These effects are found in the 'p1_pokemon_state_effects' column,
    which can contain multiple effects joined by '_'.
    """
    print("Creating trapping (clamp, wrap, firespin) features...")

    # 1. Define the trapping effects and the regex pattern
    TRAPPING_EFFECTS = ['clamp', 'wrap', 'firespin']
    # Create a regex pattern: 'clamp|wrap|firespin'
    pattern = '|'.join(TRAPPING_EFFECTS)

    # 2. Find turns where P1 was trapped
    # We use na=False to safely handle empty/NaN effect strings
    p1_trapped_turns = turns_df[turns_df['p1_pokemon_state_effects'].str.contains(pattern, na=False)]
    
    # 3. Count unique P1 Pokemon that were trapped
    p1_trapped_count = p1_trapped_turns.groupby('battle_id')['p1_pokemon_state_name'].nunique()
    p1_trapped_count.name = 'p1_trapped_pokemon_count'

    # 4. Find turns where P2 was trapped
    p2_trapped_turns = turns_df[turns_df['p2_pokemon_state_effects'].str.contains(pattern, na=False)]
    
    # 5. Count unique P2 Pokemon that were trapped
    p2_trapped_count = p2_trapped_turns.groupby('battle_id')['p2_pokemon_state_name'].nunique()
    p2_trapped_count.name = 'p2_trapped_pokemon_count'

    # 6. Merge, clean, and create advantage
    trapping_df = pd.concat([p1_trapped_count, p2_trapped_count], axis=1).fillna(0).astype(int)
    
    # Positive = P1 advantage = P2 got trapped more
    trapping_df['trapping_advantage'] = trapping_df['p2_trapped_pokemon_count'] - trapping_df['p1_trapped_pokemon_count']
    return trapping_df 


# 6 Deployed Team Mean Stat Features 
def _create_deployed_mean_stat_features(turns_df: pd.DataFrame, pokemon_stats_map: dict, default_pokemon_stats: dict) -> pd.DataFrame:
    """
    Calculates stat differentials based on the MEAN stats of the unique
    Pokémon deployed by each player during the 30 turns.
    
    This replaces the old 'Team Mean vs. Lead' logic.
    """
    print("Calculating deployed team mean stat advantages...")
    
    # 1. Create a Stats DataFrame from the Pokédex for easy lookup
    stats_df = pd.DataFrame.from_dict(pokemon_stats_map, orient='index')
    # Calculate aggregate stats. NOTE: Using Def+SpD for defense.
    stats_df['offense'] = stats_df['base_atk'] + stats_df['base_spa']
    stats_df['defense'] = stats_df['base_def'] + stats_df['base_spd'] 
    stats_df_agg = stats_df[['offense', 'defense', 'base_spe']] # Keep only what we need

    # 2. Get unique deployed Pokémon for P1 and P2 per battle
    p1_names_series = turns_df.groupby('battle_id')['p1_pokemon_state_name'].unique()
    p2_names_series = turns_df.groupby('battle_id')['p2_pokemon_state_name'].unique()

    # 3. Helper function to get the MEAN stats for a list of names
    
    # Pre-calculate default (fallback) stats
    default_offense = default_pokemon_stats.get('base_atk', 0) + default_pokemon_stats.get('base_spa', 0)
    default_defense = default_pokemon_stats.get('base_def', 0) + default_pokemon_stats.get('base_spd', 0)
    default_spe = default_pokemon_stats.get('base_spe', 0)

    def get_avg_stats(name_list, stats_lookup_df):
        valid_names = [name for name in name_list if name in stats_lookup_df.index]
        
        if not valid_names:
            # No valid Pokémon were deployed, return default stats
            return pd.Series({'offense': default_offense, 'defense': default_defense, 'base_spe': default_spe})
        
        # Look up stats for all valid names and calculate the MEAN
        return stats_lookup_df.loc[valid_names].mean()

    # 4. Apply the helper to get mean deployed stats for each battle
    p1_avg_stats = p1_names_series.apply(get_avg_stats, args=(stats_df_agg,))
    p2_avg_stats = p2_names_series.apply(get_avg_stats, args=(stats_df_agg,))

    # Rename columns for clarity before merging
    p1_avg_stats = p1_avg_stats.add_prefix('p1_deployed_mean_')
    p2_avg_stats = p2_avg_stats.add_prefix('p2_deployed_mean_')
    
    # 5. Combine and calculate final features
    final_df = pd.concat([p1_avg_stats, p2_avg_stats], axis=1)
    
    # Handle 0-division safety
    final_df['p1_deployed_mean_defense'] = final_df['p1_deployed_mean_defense'].replace(0, 1.0)
    final_df['p2_deployed_mean_defense'] = final_df['p2_deployed_mean_defense'].replace(0, 1.0)

    # --- Create features based on the old logic, but now symmetric ---
    
    # Feature 1: Speed Advantage (Deployed Mean vs. Deployed Mean)
    final_df['deployed_spe_advantage'] = final_df['p1_deployed_mean_base_spe'] - final_df['p2_deployed_mean_base_spe']
    
    # Feature 2: P1 Offense vs P2 Defense (Deployed Mean vs. Deployed Mean)
    final_df['deployed_p1_off_vs_p2_def_ratio'] = final_df['p1_deployed_mean_offense'] / final_df['p2_deployed_mean_defense']
    
    # Feature 3: Symmetric counterpart (P2 Offense vs P1 Defense)
    final_df['deployed_p2_off_vs_p1_def_ratio'] = final_df['p2_deployed_mean_offense'] / final_df['p1_deployed_mean_defense']

    # 6. Return only the final features
    return final_df[['deployed_spe_advantage', 'deployed_p1_off_vs_p2_def_ratio', 'deployed_p2_off_vs_p1_def_ratio']]



# 7. Dynamic Matchup Features: STAB and Effectiveness
def _get_types_list(row, prefix): 
    # Helper to get a list of types from one-hot encoded columns
    return [t.replace(prefix, '').lower() for t in POKEMON_TYPES if f"{prefix}{t}" in row and row[f"{prefix}{t}"] == 1]

def _create_dynamic_matchup_features(
    turns_df: pd.DataFrame, 
    teams_df: pd.DataFrame, 
    pokemon_stats_map: dict, 
    default_pokemon_stats: dict
) -> pd.DataFrame:
    
    """
    Calculates symmetric matchup features for both players:
    - STAB counts for P1 and P2
    - Effectiveness scores for P1 vs P2 and P2 vs P1
    """
    print("Creating SYMMETRIC dynamic matchup features...")
    temp_df = turns_df.copy()
    
    # --- P1 Type Logic ---
    p1_type_cols = [f'type_{t}' for t in POKEMON_TYPES]
    p1_types_df = teams_df[['battle_id', 'name'] + p1_type_cols].copy()
    p1_types_df['p1_types'] = p1_types_df[p1_type_cols].apply(lambda row: _get_types_list(row, 'type_'), axis=1 )
    p1_types_map = p1_types_df.groupby('battle_id').apply(
        lambda x: pd.Series(x['p1_types'].values, index=x['name']).to_dict(), include_groups=False).to_dict()

    temp_df['p1_current_types'] = temp_df.apply(
        lambda row: p1_types_map.get(row['battle_id'], {}).get(row['p1_pokemon_state_name'], []), axis=1)
    
    # --- P2 Type Logic ---
    temp_df['p2_current_types'] = temp_df['p2_pokemon_state_name'].astype(str).apply(
        lambda x: pokemon_stats_map.get(x, default_pokemon_stats)['types']
    )
    
    # --- Computation Flag/Score ---
    temp_df['p1_move_type_clean'] = temp_df['p1_move_details_type'].astype('object').fillna('').astype(str).str.lower()
    temp_df['p2_move_type_clean'] = temp_df['p2_move_details_type'].astype('object').fillna('').astype(str).str.lower()

    # P1 STAB 
    temp_df['p1_stab_flag'] = temp_df.apply(
        lambda row: 1 
        if (row['p1_move_type_clean'] in row['p1_current_types']) and (row['p1_move_details_category'] not in ['status', 'MISSING_MOVE']) 
        else 0, axis=1)
    
    # P2 STAB 
    temp_df['p2_stab_flag'] = temp_df.apply(
        lambda row: 1 
        if (row['p2_move_type_clean'] in row['p2_current_types']) and (row['p2_move_details_category'] not in ['status', 'MISSING_MOVE']) 
        else 0, axis=1)
        
    # P1 Efficacia vs P2 
    temp_df['p1_eff_score'] = temp_df.apply(
        lambda row: _get_type_multiplier(row['p1_move_type_clean'], row['p2_current_types']) 
        if (row['p1_move_details_category'] not in ['status', 'MISSING_MOVE'] and row['p2_current_types']) 
        else 1.0, axis=1)
    
    # P2 Efficacia vs P1 
    temp_df['p2_eff_score'] = temp_df.apply(
        lambda row: _get_type_multiplier(row['p2_move_type_clean'], row['p1_current_types']) 
        if (row['p2_move_details_category'] not in ['status', 'MISSING_MOVE'] and row['p1_current_types']) 
        else 1.0, axis=1)
    
    agg_dict = {
        'p1_stab_flag': 'sum',
        'p2_stab_flag': 'sum', # NUOVO
        'p1_eff_score': [lambda x: (x > 1).sum(), lambda x: (x < 1).sum()], # NUOVO
        'p2_eff_score': [lambda x: (x > 1).sum(), lambda x: (x < 1).sum()]
    }
    
    dynamic_matchup_df = temp_df.groupby('battle_id').agg(agg_dict)
    
    # colums names
    dynamic_matchup_df.columns = [
        'p1_stab_count',
        'p2_stab_count',
        'p1_hits_p2_super_effective', 
        'p1_hits_p2_not_effective',
        'p2_hits_p1_super_effective', 
        'p2_hits_p1_not_effective'
    ]
    return dynamic_matchup_df


# 8. Satuts Pressure
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


# 9. Realized Damage Features
def _create_realized_damage_features(
    turns_df: pd.DataFrame, 
    pokemon_stats_map: dict, 
    default_pokemon_stats: dict
) -> pd.DataFrame:
    """
    Calculates the total REALIZED damage dealt by each player,
    based on the opponent's HP delta to filter out "phantom damage".
    
    Returns a single feature: the net damage advantage for P1.
    """
    print("Creating total REALIZED damage (HP-delta) features...")
    
    # ... (Passaggi 1, 2, 3 identici a prima) ...
    # 1. Create flat stat maps from Pokédex
    avg_stats = default_pokemon_stats
    stat_maps = {
        'atk': {name: stats['base_atk'] for name, stats in pokemon_stats_map.items()},
        'def': {name: stats['base_def'] for name, stats in pokemon_stats_map.items()},
        'spa': {name: stats['base_spa'] for name, stats in pokemon_stats_map.items()},
        'spd': {name: stats['base_spd'] for name, stats in pokemon_stats_map.items()}
    }
    default_vals = {
        'atk': avg_stats.get('base_atk', 100.0),
        'def': avg_stats.get('base_def', 100.0),
        'spa': avg_stats.get('base_spa', 100.0),
        'spd': avg_stats.get('base_spd', 100.0)
    }

    # 2. Map stats to every turn
    temp_df = turns_df.copy()
    for stat in ['atk', 'def', 'spa', 'spd']:
        temp_df[f'p1_current_{stat}'] = temp_df['p1_pokemon_state_name'].astype(str).map(stat_maps[stat]).fillna(default_vals[stat])
        temp_df[f'p2_current_{stat}'] = temp_df['p2_pokemon_state_name'].astype(str).map(stat_maps[stat]).fillna(default_vals[stat])

    # 3. Determine Attacker/Defender stats
    # --- P1 attacking P2 ---
    p1_atk_stat = np.where(temp_df['p1_move_details_category'] == 'PHYSICAL', temp_df['p1_current_atk'], temp_df['p1_current_spa'])
    p2_def_stat = np.where(temp_df['p1_move_details_category'] == 'PHYSICAL', temp_df['p2_current_def'], temp_df['p2_current_spd'])
    # --- P2 attacking P1 ---
    p2_atk_stat = np.where(temp_df['p2_move_details_category'] == 'PHYSICAL', temp_df['p2_current_atk'], temp_df['p2_current_spa'])
    p1_def_stat = np.where(temp_df['p2_move_details_category'] == 'PHYSICAL', temp_df['p1_current_def'], temp_df['p1_current_spd'])
    
    # 4. "Delta HP" Check
    p1_hp_prev = temp_df.groupby('battle_id')['p1_pokemon_state_hp_pct'].shift(1).fillna(1.0)
    p2_hp_prev = temp_df.groupby('battle_id')['p2_pokemon_state_hp_pct'].shift(1).fillna(1.0)
    p1_hp_delta = temp_df['p1_pokemon_state_hp_pct'] - p1_hp_prev
    p2_hp_delta = temp_df['p2_pokemon_state_hp_pct'] - p2_hp_prev
    p1_hit_flag = np.where(p2_hp_delta < 0, 1, 0)
    p2_hit_flag = np.where(p1_hp_delta < 0, 1, 0)
    
    # 5. Calculate potential damage per turn
    p1_move_power = temp_df['p1_move_details_base_power'].fillna(0)
    p2_move_power = temp_df['p2_move_details_base_power'].fillna(0)
    
    # --- INIZIO BLOCCO CORRETTO ---
    # Sostituiamo .replace(0, 1) con np.where(..., 1.0, ...)
    p2_def_stat_safe = np.where(p2_def_stat == 0, 1.0, p2_def_stat)
    p1_def_stat_safe = np.where(p1_def_stat == 0, 1.0, p1_def_stat)

    p1_turn_potential_damage = (p1_atk_stat * p1_move_power) / p2_def_stat_safe
    p2_turn_potential_damage = (p2_atk_stat * p2_move_power) / p1_def_stat_safe
    # --- FINE BLOCCO CORRETTO ---
    
    # 6. Apply "Realized Damage" flags
    temp_df['p1_realized_damage'] = p1_turn_potential_damage * p1_hit_flag
    temp_df['p2_realized_damage'] = p2_turn_potential_damage * p2_hit_flag

    # 7. Aggregate total damage per battle
    damage_df = temp_df.groupby('battle_id')[['p1_realized_damage', 'p2_realized_damage']].sum()
    
    # 8. Create advantage feature
    damage_df['realized_damage_advantage'] = damage_df['p1_realized_damage'] - damage_df['p2_realized_damage']
    
    # 9. Return only the final advantage feature
    return damage_df
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------- Main feature engineering function -----------------------------------------------
def feature_engineering_version_14(
    train: bool,
    battles_df: pd.DataFrame, 
    turns_df: pd.DataFrame, 
    teams_df: pd.DataFrame,
    pokemon_stats_map: dict,  
    default_pokemon_stats: dict
) -> pd.DataFrame:
    
    final_df = battles_df.copy()

    # 1. KO and HP features 
    print("Creating timeline feature for ko and hp%...")
    timeline_features = _create_timeline_features(turns_df, pokemon_stats_map, default_pokemon_stats)
    final_df = pd.merge(final_df, timeline_features, on='battle_id', how='left')

    # 2. Status features
    print("Creating status features...")
    status_features = _create_status_features(turns_df)
    final_df = pd.merge(final_df, status_features, on='battle_id', how='left')

    # 3. Switch Pressure features
    print("Creating Switch Pressure features...")
    switch_features = _create_switch_pressure_features(turns_df)
    final_df = pd.merge(final_df, switch_features, on='battle_id', how='left')

    # 4. Last Turn Boost Advantage features
    print("Creating last turn boost advantage features...")
    boost_features = _create_last_turn_boost_advantage(turns_df)
    final_df = pd.merge(final_df, boost_features, on='battle_id', how='left')

    # 5. Trapping features
    print("Creating Trapping features...")
    trapping_features = _create_trapping_features(turns_df)
    final_df = pd.merge(final_df, trapping_features, on='battle_id', how='left')

    # 6. Deployed Mean Stat Features
    print("Creating deployed mean stat features...")
    deployed_mean_stat_features = _create_deployed_mean_stat_features(turns_df, pokemon_stats_map, default_pokemon_stats)
    final_df = pd.merge(final_df, deployed_mean_stat_features, on='battle_id', how='left')

    # 7. Dynamic Matchup Features# 4. Dynamic Matchup Features
    print("Creating dynamic matchup features...")
    dynamic_matchup_features = _create_dynamic_matchup_features(turns_df, teams_df, pokemon_stats_map, default_pokemon_stats)
    final_df = pd.merge(final_df, dynamic_matchup_features, on='battle_id', how='left')

    # 8. Status Pressure Features
    print("Creating status pressure features...")
    status_pressure_features = _create_status_pressure_features(turns_df)
    final_df = pd.merge(final_df, status_pressure_features, on='battle_id', how='left')

    # 9. Realized Damage Features
    print("Creating realized damage features...")
    realized_damage_features = _create_realized_damage_features(turns_df, pokemon_stats_map, default_pokemon_stats)
    final_df = pd.merge(final_df, realized_damage_features, on='battle_id', how='left')


    # ------------------------- Final Cleanup --------------------------
    lead_types_col = ['p1_lead_types', 'p2_lead_types'] 
    p2_types_cols = [f'p2_lead_type_{t}' for t in POKEMON_TYPES]
    redundant_component_features = [
        # Type components
        'p1_type_matchup_score', 'p2_type_matchup_score',
        
        # p1 Stat Components
        'p1_team_mean_atk', 
        'p1_team_mean_spa', 'p1_team_mean_spe',
        'p1_mean_offense', 'p1_mean_defense',
        
        # p2 Lead Stat Components
        'p2_lead_base_def',
        'p2_lead_base_spa', 'p2_lead_base_spe',
        'p2_lead_defense',
        'p2_lead_name', 'p2_lead_base_hp',
        'p2_lead_base_atk', 'p2_lead_offense','p2_lead_base_spd',
        'p2_lead_level',  

        # features
        'p1_ko_count', 'p2_ko_count', 'ko_advantage',

        'psn_advantage', 'tox_advantage', 'slp_advantage', 'frz_advantage','par_advantage','brn_advantage',
        #'p1_brn_count', 'p2_brn_count', 'p1_par_count', 'p2_par_count', 
        #'p1_psn_count', 'p2_psn_count', 'p1_tox_count', 'p2_tox_count', 
        #'p1_slp_count', 'p2_slp_count', 'p1_frz_count', 'p2_frz_count', 'brn_advantage' 

        'switch_advantage',
        #'p1_switch_count', 'p2_switch_count',

        'trapping_advantage',
        #'p1_trapped_pokemon_count', 'p2_trapped_pokemon_count' 

        'realized_damage_advantage',
        #'p1_realized_damage','p2_realized_damage'


        
    ]
    
    # Define columns to drop: temp, original P2 types, and redundant components
    cols_to_drop = lead_types_col + p2_types_cols + redundant_component_features
    final_df = final_df.drop(columns=cols_to_drop, errors='ignore').fillna(0)

    # Ensure correct dtypes
    final_df = final_df.infer_objects(copy=False)
    print(f"Feature engineering version 14 completed. Feature count: {final_df.shape[1]}")
    return final_df
    
