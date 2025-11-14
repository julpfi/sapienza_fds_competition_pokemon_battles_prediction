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
            multiplier *= type_chart.get(move_type, {}).get(target_type, 1.0)
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


def _count_strategic_switches(df: pd.DataFrame, player: str) -> pd.Series:
        cols = [f'{player}_pokemon_state_name', f'{player}_pokemon_state_status']

        # Sort to ensure corrcet order 
        sort_cols = ['battle_id'] + (['turn'] if 'turn' in df.columns else [])
        df_switch = df.sort_values(sort_cols)[['battle_id'] + cols].copy()

        # Compare to previous turn
        df_switch['prev_name'] = df_switch.groupby('battle_id')[cols[0]].shift(1)
        df_switch['prev_status'] = df_switch.groupby('battle_id')[cols[1]].shift(1)

        # Voluntary switch = name changed AND previous pokemon not fainted
        df_switch['is_switch'] = (
            (df_switch[cols[0]] != df_switch['prev_name']) &
            (df_switch['prev_status'] != 'fnt') &
            df_switch['prev_name'].notna()
        )

        return df_switch.groupby('battle_id')['is_switch'].sum().astype(int)


def _predicted_damage(row, side):
    if row.get(f"{side}_move_details_category") == "PHYSICAL":
        atk = row.get(f"{side}_lead_base_atk", 0)
        df_def = row.get(f'{"p2" if side=="p1" else "p1"}_lead_base_def', 1)
    elif row.get(f"{side}_move_details_category") == "SPECIAL":
        atk = row.get(f"{side}_lead_base_spa", 0)
        df_def = row.get(f'{"p2" if side=="p1" else "p1"}_lead_base_spd', 1)
    else:
        return 0.0

    base_power = row.get(f"{side}_move_details_base_power", 0) or 0
    acc = row.get(f"{side}_move_details_accuracy", 1.0) or 1.0
    move_type = str(row.get(f"{side}_move_details_type", "")).lower()
    types = row.get(f"{side}_lead_types", [])
    stab = 1.5 if move_type in [t.lower() for t in types] else 1.0

    return base_power * (atk / max(df_def, 1)) * acc * stab


# ---------------------------------------------------------------------------------------------------------------
# -------------------------------------- Feature Creators ------------------------------------------------------

# 1. P1 Team Stat Features
def _create_team_stat_features(teams_df: pd.DataFrame) -> pd.DataFrame:
    # Creates aggregated stat features for P1's team
    team_stats_cols = [f"base_{s}" for s in ["hp", "atk", "def", "spa", "spd", "spe"]]
    team_features = teams_df.groupby('battle_id')[team_stats_cols].mean().reset_index()
    
    # Rename columns for clarity
    team_features.columns = ['battle_id'] + [f'p1_team_mean_{col.replace("base_", "")}' for col in team_stats_cols]
    
    return team_features

# 2. Timeline Features - KO counts and HP% dynamics 
def _create_timeline_features_ko(turns_df: pd.DataFrame) -> pd.DataFrame:
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

    # KO Advantage: Positive means P1 has an advantage (P2 has more KOs)
    ko_df['ko_advantage'] = ko_df['p2_ko_count'] - ko_df['p1_ko_count']
    return ko_df


# 2 Timlime Features: KO and HP% 
def _create_timeline_features(turns_df: pd.DataFrame) -> pd.DataFrame:
    # Creates aggregated KO and HP % features from the timeline.
    # Includes both team average HP (relative) and team sum HP (total) as well as ko counts and ko advantage

    
    # --- HP % ---
    
    # 1. Get the last HP% for every pokemon that participated 
    p1_last_hp_per_pokemon = turns_df.groupby(['battle_id', 'p1_pokemon_state_name'], observed=False)['p1_pokemon_state_hp_pct'].last()
    p2_last_hp_per_pokemon = turns_df.groupby(['battle_id', 'p2_pokemon_state_name'], observed=False)['p2_pokemon_state_hp_pct'].last()
    
    # 2. Calculate average hp % 
    p1_team_avg_hp = p1_last_hp_per_pokemon.groupby('battle_id').mean()
    p1_team_avg_hp.name = 'p1_team_avg_hp'
    p2_team_avg_hp = p2_last_hp_per_pokemon.groupby('battle_id').mean()
    p2_team_avg_hp.name = 'p2_team_avg_hp'
    
    # 3. Calculate total hp %     
    # --- P1 SUM HP ---
    p1_participating_sum_hp = p1_last_hp_per_pokemon.groupby('battle_id').sum()
    p1_participating_count = p1_last_hp_per_pokemon.groupby('battle_id').size()
    p1_non_participating_hp = (6 - p1_participating_count) * 1.0
    p1_team_sum_hp = (p1_participating_sum_hp + p1_non_participating_hp)
    p1_team_sum_hp.name = 'p1_team_sum_hp'

    # --- P2 SUM HP ---
    p2_participating_sum_hp = p2_last_hp_per_pokemon.groupby('battle_id').sum()
    p2_participating_count = p2_last_hp_per_pokemon.groupby('battle_id').size()
    p2_non_participating_hp = (6 - p2_participating_count) * 1.0
    p2_team_sum_hp = (p2_participating_sum_hp + p2_non_participating_hp)
    p2_team_sum_hp.name = 'p2_team_sum_hp'



    # 4. Merge HP features 
    hp_df = pd.merge(p1_team_avg_hp, p2_team_avg_hp, on='battle_id', how='outer')
    hp_df = pd.merge(hp_df, p1_team_sum_hp, on='battle_id', how='outer')
    hp_df = pd.merge(hp_df, p2_team_sum_hp, on='battle_id', how='outer')
    
    # 5. FillNa 
    hp_df['p1_team_avg_hp'] = hp_df['p1_team_avg_hp'].fillna(0.5) # (neutral)
    hp_df['p2_team_avg_hp'] = hp_df['p2_team_avg_hp'].fillna(0.5) # (neutral)
    
    # If sum_hp is NaN, it means 0 participants. 
    # The sum for both should be 6.0 (6 Pokemon at 100% HP).
    hp_df['p1_team_sum_hp'] = hp_df['p1_team_sum_hp'].fillna(6.0) 
    hp_df['p2_team_sum_hp'] = hp_df['p2_team_sum_hp'].fillna(6.0) 

    # 6. Create advantage features 
    hp_df['team_hp_advantage'] = hp_df['p1_team_avg_hp'] - hp_df['p2_team_avg_hp']
    hp_df['team_hp_sum_advantage'] = hp_df['p1_team_sum_hp'] - hp_df['p2_team_sum_hp']
    

    return hp_df

def _create_timeline_features_hp(turns_df: pd.DataFrame, teams_df: pd.DataFrame, battles_df: pd.DataFrame) -> pd.DataFrame: 
    # Get HP stats for all pokemon  (participated in turns)
    hp_stats = turns_df.groupby(['battle_id', 'p1_pokemon_state_name'], observed=False)['p1_pokemon_state_hp_pct'].agg(['first', 'last', 'min'])
    p1_hp_first = hp_stats['first']
    p1_hp_last = hp_stats['last']
    p1_hp_min = hp_stats['min']

    hp_stats_p2 = turns_df.groupby(['battle_id', 'p2_pokemon_state_name'], observed=False)['p2_pokemon_state_hp_pct'].agg(['first', 'last', 'min'])
    p2_hp_first = hp_stats_p2['first']
    p2_hp_last = hp_stats_p2['last']
    p2_hp_min = hp_stats_p2['min']

    # Measures the maximum potential damage pressure  
    p1_max_damage_per_mon = (p1_hp_first - p1_hp_min).clip(lower=0)
    p2_max_damage_per_mon = (p2_hp_first - p2_hp_min).clip(lower=0)

    p1_damage_pressure = p1_max_damage_per_mon.groupby('battle_id').mean()
    p2_damage_pressure = p2_max_damage_per_mon.groupby('battle_id').mean()

    pressure_advantage = (p2_damage_pressure - p1_damage_pressure).fillna(0)
    pressure_advantage.name = 'damage_pressure_advantage'

    # Total hp loss 
    p1_total_hp_loss = (p1_hp_first - p1_hp_last).clip(lower=0).groupby('battle_id').sum()
    p2_total_hp_loss = (p2_hp_first - p2_hp_last).clip(lower=0).groupby('battle_id').sum()

    total_loss_advantage = (p2_total_hp_loss - p1_total_hp_loss).fillna(0)
    total_loss_advantage.name = 'hp_total_loss_advantage'

    # Momentum: Compare early and late battle averages 
    max_turn = turns_df.groupby('battle_id')['turn'].max()
    turn_center_cutoff = (max_turn / 2).astype(int)

        # Early battle
    early_turns = turns_df.merge(turn_center_cutoff.rename('cutoff'), left_on='battle_id', right_index=True)
    early_turns = early_turns[early_turns['turn'] <= early_turns['cutoff']]

    p1_early_hp = early_turns.groupby('battle_id')['p1_pokemon_state_hp_pct'].mean()
    p2_early_hp = early_turns.groupby('battle_id')['p2_pokemon_state_hp_pct'].mean()

        # Late battle
    late_turns = turns_df.merge(turn_center_cutoff.rename('cutoff'), left_on='battle_id', right_index=True)
    late_turns = late_turns[late_turns['turn'] > late_turns['cutoff']]

    p1_late_hp = late_turns.groupby('battle_id')['p1_pokemon_state_hp_pct'].mean()
    p2_late_hp = late_turns.groupby('battle_id')['p2_pokemon_state_hp_pct'].mean()

        # Momentum = late HP - early HP
    p1_momentum = (p1_late_hp - p1_early_hp).fillna(0)
    p2_momentum = (p2_late_hp - p2_early_hp).fillna(0)

    momentum_advantage = (p1_momentum - p2_momentum).fillna(0)
    momentum_advantage.name = 'hp_momentum_advantage'

    # HP trend 
    hp_trend = turns_df.groupby('battle_id').apply(
        lambda df: np.polyfit(df['turn'], df['p1_pokemon_state_hp_pct'] - df['p2_pokemon_state_hp_pct'], 1)[0])
    hp_trend.name = 'hp_trend_slope'


    # Healthy pokemons in backup (> 50%)
    p1_full_team = teams_df[['battle_id', 'name']].rename(columns={'name': 'p1_pokemon_state_name'})
    p1_full_team = p1_full_team.set_index(['battle_id', 'p1_pokemon_state_name'])

    p1_hp_last = p1_hp_last.rename('p1_pokemon_state_hp_pct')

    p1_full_team_hp = p1_full_team.merge(p1_hp_last, left_index=True, right_index=True, how='left')['p1_pokemon_state_hp_pct'].fillna(1.0)
      
    p1_healthy = (p1_full_team_hp > 0.5).groupby('battle_id').sum()
    p2_healthy = (p2_hp_last > 0.5).groupby('battle_id').sum()

    reserves_advantage = (p1_healthy - p2_healthy).fillna(0).astype(int)
    reserves_advantage.name = 'healthy_pokemon_advantage'

    # Combine all hp features 
    hp_features = pd.DataFrame({
    'damage_pressure_advantage': pressure_advantage,
    'hp_total_loss_advantage': total_loss_advantage,
    'hp_trend_slope': hp_trend,
    'hp_momentum_advantage': momentum_advantage,
    'healthy_pokemon_advantage': reserves_advantage    
    }).fillna(0)

    return hp_features


# 3 Status Pressure Features
def _create_status_pressure_features(turns_df: pd.DataFrame) -> pd.DataFrame:
    # Calculates total turns spent afflicted by status, differentiating by severity

    # "Major" status (annoying, but you can still move)
    MAJOR_STATUS = ['par', 'brn', 'psn', 'tox']
    # "Critical" status (move-ending, lose your turn)
    CRITICAL_STATUS = ['slp', 'frz']
    CRITICAL_STATUS_MOVES = {'hypnosis','spore','sing','icebeam','blizzard'} 
    
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

    # 3. Critical status success rate 
    p1_attempts = turns_df[turns_df['p1_move_details_name'].isin(CRITICAL_STATUS_MOVES)].groupby('battle_id').size()
    p1_attempts.name = 'p1_attempts'
    p1_success = turns_df[turns_df['p2_pokemon_state_status'].isin(CRITICAL_STATUS) & turns_df['p1_move_details_name'].isin(CRITICAL_STATUS_MOVES)].groupby('battle_id').size()
    p1_success.name = 'p1_success' 

    p2_attempts = turns_df[turns_df['p2_move_details_name'].isin(CRITICAL_STATUS_MOVES)].groupby('battle_id').size()
    p2_attempts.name = 'p2_attempts'
    p2_success = turns_df[turns_df['p1_pokemon_state_status'].isin(CRITICAL_STATUS) & turns_df['p2_move_details_name'].isin(CRITICAL_STATUS_MOVES)].groupby('battle_id').size()
    p2_success.name = 'p2_success' 

    success_rate_df = pd.concat([p1_attempts, p1_success, p2_attempts, p2_success], axis=1).fillna(0)

    success_rate_df['p1_critical_status_success_rate'] = success_rate_df['p1_success'] / success_rate_df['p1_attempts'].replace(0, np.nan)
    success_rate_df['p2_critical_status_success_rate'] = success_rate_df['p2_success'] / success_rate_df['p2_attempts'].replace(0, np.nan)

    final_rates_df = success_rate_df[['p1_critical_status_success_rate', 'p2_critical_status_success_rate']]

    # 4. Merge all features
    status_df = pd.merge(p1_major_status_turns, p2_major_status_turns, on='battle_id', how='outer')
    status_df = pd.merge(status_df, p1_critical_status_turns, on='battle_id', how='outer')
    status_df = pd.merge(status_df, p2_critical_status_turns, on='battle_id', how='outer')
    status_df = pd.merge(status_df, final_rates_df, on='battle_id', how='outer')

    status_df = status_df.fillna(0)    
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

    temp_df['p1_current_types'] = temp_df.apply(lambda row: p1_types_map.get(row['battle_id'], {}).get(row['p1_pokemon_state_name'], []), axis=1)
    
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
        'p2_eff_score': [('p2_hits_p1_super_effective', lambda x: (x > 1).sum()),
                        ('p2_hits_p1_not_effective', lambda x: (x < 1).sum())]
        }

    
    dynamic_matchup_df = temp_df.groupby('battle_id').agg(agg_dict)
    
    dynamic_matchup_df.columns = [
        'p1_stab_count',
        'p2_hits_p1_super_effective', 
        'p2_hits_p1_not_effective'
        ]
    return dynamic_matchup_df


# 5. Lead Differentials Features 
def _create_lead_differential_features(battles_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:

    stats = ["hp", "atk", "def", "spa", "spd", "spe"]

    # p1 team mean stats
    p1_team_stats_cols = [f"base_{s}" for s in stats]
    team_features_df = teams_df.groupby('battle_id')[p1_team_stats_cols].mean()
    team_features_df.columns = [f'p1_team_mean_{s}' for s in stats]
    team_features_df['p1_mean_offense'] = team_features_df['p1_team_mean_atk'] + team_features_df['p1_team_mean_spa']

    # p1 lead
    p1_lead_cols = ['battle_id'] + [f"base_{s}" for s in stats]
    p1_lead_df = teams_df[teams_df['pokemon_nr'] == 0][p1_lead_cols].copy()
    p1_rename_map = {f"base_{s}": f"p1_lead_base_{s}" for s in stats}
    p1_lead_df = p1_lead_df.rename(columns=p1_rename_map)

    # p2 lead
    p2_lead_cols = ['battle_id'] + [f"p2_lead_base_{s}" for s in stats]
    p2_lead_df = battles_df[p2_lead_cols].copy()

    diff_df = pd.merge(p2_lead_df, p1_lead_df, on='battle_id', how='left')
    diff_df = pd.merge(diff_df, team_features_df, on='battle_id', how='left')

    all_stat_cols = [col for col in diff_df.columns if 'base_' in col or 'mean_' in col]
    diff_df[all_stat_cols] = diff_df[all_stat_cols].fillna(0)

    # Lead stat differentials        
    for s in stats:
        diff_df[f'lead_diff_{s}'] = diff_df[f'p1_lead_base_{s}'] - diff_df[f'p2_lead_base_{s}']
    
    p1_offense = diff_df['p1_lead_base_atk'] + diff_df['p1_lead_base_spa']
    p2_offense = diff_df['p2_lead_base_atk'] + diff_df['p2_lead_base_spa']
    diff_df['lead_diff_offense'] = p1_offense - p2_offense

    p1_defense = diff_df['p1_lead_base_def'] + diff_df['p1_lead_base_spd']
    p2_defense = diff_df['p2_lead_base_def'] + diff_df['p2_lead_base_spd']
    diff_df['lead_diff_defense'] = p1_defense - p2_defense
    
    # Total defense of p2 lead
    diff_df['p2_lead_defense_total'] = diff_df['p2_lead_base_def'] + diff_df['p2_lead_base_spd']
    
    # Speed advantage of p1 mean vs p2 lead
    diff_df['lead_team_mean_spe_adv'] = diff_df['p1_team_mean_spe'] - diff_df['p2_lead_base_spe']
    
    # Offensive power ratio
    diff_df['p1_team_off_vs_p2_lead_def_ratio'] = diff_df['p1_mean_offense'] / diff_df['p2_lead_defense_total'].replace(0, 1)

    lead_diff_cols = [f'lead_diff_{s}' for s in stats] + ['lead_diff_offense', 'lead_diff_defense']
    team_vs_lead_cols = ['lead_team_mean_spe_adv', 'p1_team_off_vs_p2_lead_def_ratio']
    
    final_cols = ['battle_id'] + lead_diff_cols + team_vs_lead_cols
    return diff_df[final_cols].fillna(0)


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
    
    # TODO Extend type dimensions? 

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


# 9. Boost Advantage Features
def _create_boost_advantage_features(turns_df: pd.DataFrame) -> pd.DataFrame:
    # Calculates the final sum of stat boosts for each team -> Relates to setup of sweeper pokemons     
    boost_stats = ['atk', 'def', 'spa', 'spd', 'spe']
    p1_boost_cols = [f'p1_pokemon_state_boost_{s}' for s in boost_stats]
    p2_boost_cols = [f'p2_pokemon_state_boost_{s}' for s in boost_stats]
    
    turns_df[p1_boost_cols] = turns_df[p1_boost_cols].fillna(0)
    turns_df[p2_boost_cols] = turns_df[p2_boost_cols].fillna(0)

    # Only consider the last boost state for each pokemon in each battle (before swapped out or fainted)
    p1_last_boosts = turns_df.groupby(['battle_id', 'p1_pokemon_state_name'], observed=False)[p1_boost_cols].last()
    p2_last_boosts = turns_df.groupby(['battle_id', 'p2_pokemon_state_name'], observed=False)[p2_boost_cols].last()

    # Sum these final boosts across all pokemon on the team
    p1_total_boosts = p1_last_boosts.groupby('battle_id').sum()
    p2_total_boosts = p2_last_boosts.groupby('battle_id').sum()
    
    # Rename columns 
    p1_total_boosts.columns = [f'p1_team_boost_sum_{s}' for s in boost_stats]
    p2_total_boosts.columns = [f'p2_team_boost_sum_{s}' for s in boost_stats]

    # Merge boosts of both players 
    boost_df = pd.merge(p1_total_boosts, p2_total_boosts, on='battle_id', how='outer')
    boost_df = boost_df.fillna(0).astype(int)

    # Overall boost advantages:     
    # Offensive advantage
    p1_offense_boost = boost_df['p1_team_boost_sum_atk'] + boost_df['p1_team_boost_sum_spa']
    p2_offense_boost = boost_df['p2_team_boost_sum_atk'] + boost_df['p2_team_boost_sum_spa']
    boost_df['offense_boost_advantage'] = p1_offense_boost - p2_offense_boost

    # Defensive advantage
    p1_defense_boost = boost_df['p1_team_boost_sum_def'] + boost_df['p1_team_boost_sum_spd']
    p2_defense_boost = boost_df['p2_team_boost_sum_def'] + boost_df['p2_team_boost_sum_spd']
    boost_df['defense_boost_advantage'] = p1_defense_boost - p2_defense_boost
    
    # Speed advantage
    boost_df['speed_boost_advantage'] = boost_df['p1_team_boost_sum_spe'] - boost_df['p2_team_boost_sum_spe']
    
    cols_to_drop = [col for col in boost_df.columns if 'p1_team_boost_sum' in col or 'p2_team_boost_sum' in col]
    boost_df = boost_df.drop(columns=cols_to_drop)
    return boost_df


# 11. Big Three Features
def _create_big_three_features(teams_df: pd.DataFrame, turns_df: pd.DataFrame) -> pd.DataFrame:
    # Checks for the presence of the "Big Three" (=meta) pokemon of Gen1 which are Tauros, Snorlax, Chansey 
    '''
    Reference
    https://gamefaqs.gamespot.com/boards/198314-pokemon-yellow-version-special-pikachu-edition/80535752
    https://www.reddit.com/r/stunfisk/comments/14kq2gu/who_are_the_big_three_of_each_gen_of_ou/?tl=de#:~:text=Ich%20w%C3%BCrde%20pers%C3%B6nlich%20Latios/Ferrothorn/Excadrill%20w%C3%A4hlen%2C%20aber%20ich,Jahren%20f%C3%BCr%20ORAS%20gefragt%20h%C3%A4tte%2C%20w%C3%BCrde%20man
    '''
    
    BIG_THREE = ['tauros', 'snorlax', 'chansey']
    
    p1_teams = teams_df[teams_df['name'].isin(BIG_THREE)]
    # Three bools for presence of each pokemon in p1's team
    p1_has_chansey = (p1_teams['name'] == 'chansey').groupby(p1_teams['battle_id']).any()
    p1_has_snorlax = (p1_teams['name'] == 'snorlax').groupby(p1_teams['battle_id']).any()
    p1_has_tauros = (p1_teams['name'] == 'tauros').groupby(p1_teams['battle_id']).any()

    p1_presence = pd.concat([p1_has_chansey, p1_has_snorlax, p1_has_tauros], axis=1)
    p1_presence.columns = ['p1_has_chansey', 'p1_has_snorlax', 'p1_has_tauros']

    # For p2's team, we check which pokemons were revealed during the battle
    p2_revealed_mons = turns_df.groupby('battle_id')['p2_pokemon_state_name'].unique().apply(lambda x: set() if not isinstance(x, set) else x)
    p2_revealed_mons.name = 'p2_revealed_set'
    
    p2_presence_df = pd.DataFrame(index=p2_revealed_mons.index)
    p2_presence_df['p2_has_chansey'] = p2_revealed_mons.apply(lambda x: 'chansey' in x)
    p2_presence_df['p2_has_snorlax'] = p2_revealed_mons.apply(lambda x: 'snorlax' in x)
    p2_presence_df['p2_has_tauros'] = p2_revealed_mons.apply(lambda x: 'tauros' in x)


    final_df = pd.merge(p1_presence, p2_presence_df, on='battle_id', how='outer')
    final_df = final_df.fillna(False).astype(int) 

    # Create advantage features
    final_df['chansey_adv'] = final_df['p1_has_chansey'] - final_df['p2_has_chansey']
    final_df['snorlax_adv'] = final_df['p1_has_snorlax'] - final_df['p2_has_snorlax']
    final_df['tauros_adv'] = final_df['p1_has_tauros'] - final_df['p2_has_tauros']
    
    # Create total count advantage
    p1_count = final_df['p1_has_chansey'] + final_df['p1_has_snorlax'] + final_df['p1_has_tauros']
    p2_count = final_df['p2_has_chansey'] + final_df['p2_has_snorlax'] + final_df['p2_has_tauros']
    final_df['big_three_adv'] = p1_count - p2_count

    # Interaction of having both chansey and snorlax -> Having both in team 
    final_df['p1_def_core'] = final_df['p1_has_chansey'] * final_df['p1_has_snorlax']
    final_df['p2_def_core'] = final_df['p2_has_chansey'] * final_df['p2_has_snorlax']
    
    final_df['defensive_core_adv'] = final_df['p1_def_core'] - final_df['p2_def_core']

    # Drop working columns
    cols_to_drop = ['p1_has_chansey', 'p1_has_snorlax', 'p1_has_tauros','p2_has_chansey', 'p2_has_snorlax', 'p2_has_tauros']
    final_df = final_df.drop(columns=cols_to_drop)
    
    return final_df

'''
Move features relating to core strategies of Gen1 UO meta: 
Trapping moves are used for offensive control preventing the opponent from attacking or switching for several turns. 
Recovery moves allow key pokemon to heal damage and increase long-term sustainability of the team. 
Trading moves are a so called wall-breaking tactic, sacrificing one's own pokemon to deal massive damage and remove a critical and dangerous threat from the opponent. 
'''

# 12. Trapping features
def _create_trapping_features(turns_df: pd.DataFrame) -> pd.DataFrame:
    # Counts the number of turns each player spent using a trapping move. Part of "offensive" stalling strategy. Very impactful in Gen1 OU
    
    TRAPPING_MOVES = ['bind', 'wrap', 'clamp', 'firespin']
    
    # Count turns
    p1_trap_turns = turns_df[turns_df['p1_move_details_name'].isin(TRAPPING_MOVES)].groupby('battle_id').size()
    p1_trap_turns.name = 'p1_trap_turns'
    
    p2_trap_turns = turns_df[turns_df['p2_move_details_name'].isin(TRAPPING_MOVES)].groupby('battle_id').size()
    p2_trap_turns.name = 'p2_trap_turns'

    # Merge counts
    trap_df = pd.merge(p1_trap_turns, p2_trap_turns, on='battle_id', how='outer')
    trap_df = trap_df.fillna(0).astype(int)
    
    # Create advantage feature
    trap_df['trap_turn_advantage'] = trap_df['p1_trap_turns'] - trap_df['p2_trap_turns']
    
    final_df = trap_df[['trap_turn_advantage']]
    return final_df


# 13. Recovery Moves Features
def _create_recovery_move_features(turns_df: pd.DataFrame) -> pd.DataFrame:
    # Counts the number of turns each player uses a recovery move. Part of the "defensive" stalling strategy
    
    RECOVERY_MOVES = ['recover', 'softboiled', 'rest']
    
    # Count turns
    p1_recovery_turns = turns_df[turns_df['p1_move_details_name'].isin(RECOVERY_MOVES)].groupby('battle_id').size()
    p1_recovery_turns.name = 'p1_recovery_turns'
    
    p2_recovery_turns = turns_df[turns_df['p2_move_details_name'].isin(RECOVERY_MOVES)].groupby('battle_id').size()
    p2_recovery_turns.name = 'p2_recovery_turns'

    # Merge counts
    recovery_df = pd.merge(p1_recovery_turns, p2_recovery_turns, on='battle_id', how='outer')
    recovery_df = recovery_df.fillna(0).astype(int)
    
    # Create advantage feature
    recovery_df['recovery_turn_adv'] = recovery_df['p1_recovery_turns'] - recovery_df['p2_recovery_turns']

    final_df = recovery_df[['recovery_turn_adv']]
    return final_df


# 14. Trading Moves Features 
def _create_trading_features(turns_df: pd.DataFrame) -> pd.DataFrame:
    # Counts how many times each player used a trading move. Part of core strategy to counter meta pokemons 
    
    TRADE_MOVES = ['explosion', 'selfdestruct']
    
    # Count moves 
    p1_trade_turns = turns_df[turns_df['p1_move_details_name'].isin(TRADE_MOVES)].groupby('battle_id').size()
    p1_trade_turns.name = 'p1_trade_turns'
    
    p2_trade_turns = turns_df[turns_df['p2_move_details_name'].isin(TRADE_MOVES)].groupby('battle_id').size()
    p2_trade_turns.name = 'p2_trade_turns'

    # Merge counts
    trade_df = pd.merge(p1_trade_turns, p2_trade_turns, on='battle_id', how='outer')
    trade_df = trade_df.fillna(0).astype(int)
    
    # Create advantage feature 
    trade_df['trade_turn_adv'] = trade_df['p1_trade_turns'] - trade_df['p2_trade_turns']

    # Addtition: Check for success of trade move
    turns_df['p1_explosion_success'] = ((turns_df['p1_move_details_name'].isin(TRADE_MOVES)) &(turns_df['p2_pokemon_state_status'] == 'fnt')).astype(int)
    turns_df['p2_explosion_success'] = ((turns_df['p2_move_details_name'].isin(TRADE_MOVES)) &(turns_df['p1_pokemon_state_status'] == 'fnt')).astype(int)

    p1_explode_success = turns_df.groupby('battle_id')['p1_explosion_success'].sum()
    p2_explode_success = turns_df.groupby('battle_id')['p2_explosion_success'].sum()

    trade_success_df = pd.merge(p1_explode_success, p2_explode_success, on='battle_id', how='outer').fillna(0).astype(int)
    trade_success_df['trade_success_adv'] = trade_success_df['p1_explosion_success'] - trade_success_df['p2_explosion_success']

    trade_df = pd.merge(trade_df, trade_success_df[['trade_success_adv']], on='battle_id', how='left')

    final_df = trade_df[['trade_turn_adv', 'trade_success_adv']]
    return final_df


# 15. Strategic switches 
def _create_switch_features(turns_df: pd.DataFrame) -> pd.DataFrame:
    
    p1_switch = _count_strategic_switches(turns_df, 'p1')
    p2_switch = _count_strategic_switches(turns_df, 'p2')

    switch_df = (
        pd.concat([p1_switch, p2_switch], axis=1)
        .fillna(0)
        .reset_index()
        .rename(columns={'is_switch': 'switch_count'})
    )
    switch_df.columns = ['battle_id', 'p1_switch_count', 'p2_switch_count']
    switch_df['switch_advantage'] = switch_df['p1_switch_count'] - switch_df['p2_switch_count']
    return switch_df


# 16. Feature: Predicted Damage Ratio on Turn 1
def _create_predicted_damage_turn_1(battles_df: pd.DataFrame, teams_df: pd.DataFrame, turns_df: pd.DataFrame) -> pd.DataFrame:
    
    t1 = turns_df[turns_df["turn"] == 1].copy()

    if "types" not in teams_df.columns:
        teams_df["types"] = teams_df.apply(lambda r: [str(r.get("type_1", "notype")).lower(), str(r.get("type_2", "notype")).lower()], axis=1)

    p1_lead = (teams_df[teams_df["pokemon_nr"] == 0][["battle_id", "base_atk", "base_spa", "base_def", "base_spd", "base_spe", "types"]]
        .rename(columns={
                "base_atk": "p1_lead_base_atk",
                "base_spa": "p1_lead_base_spa",
                "base_def": "p1_lead_base_def",
                "base_spd": "p1_lead_base_spd",
                "base_spe": "p1_lead_base_spe",
                "types": "p1_lead_types"}))

    p2_lead_cols = ["battle_id", "p2_lead_base_atk", "p2_lead_base_spa", "p2_lead_base_def", "p2_lead_base_spd", "p2_lead_types"]
    if "p2_lead_types" not in battles_df.columns:
        battles_df["p2_lead_types"] = battles_df.apply(lambda row: _get_types_list(row, 'p2_lead_type_'), axis=1)


    t1 = t1.merge(p1_lead, on="battle_id", how="left")
    t1 = t1.merge(battles_df[p2_lead_cols], on="battle_id", how="left")

    t1["p1_expected_dmg"] = t1.apply(lambda r: _predicted_damage(r, "p1"), axis=1)
    t1["p2_expected_dmg"] = t1.apply(lambda r: _predicted_damage(r, "p2"), axis=1)

    t1["expected_damage_ratio_turn_1"] = np.log1p(t1["p1_expected_dmg"]) - np.log1p(t1["p2_expected_dmg"])
    return t1[["battle_id", "expected_damage_ratio_turn_1"]].fillna(0)


# -------------------------------------------------------------------------------------------------------------
# ---------------------------- Main feature engineering function -----------------------------------------------
def feature_engineering_version_13(
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
    timeline_ko_features = _create_timeline_features_ko(turns_df)
    timeline_hp_features = _create_timeline_features_hp(turns_df=turns_df, teams_df=teams_df, battles_df=battles_df)
    final_df = pd.merge(final_df, timeline_ko_features, on='battle_id', how='left')
    final_df = pd.merge(final_df, timeline_hp_features, on='battle_id', how='left')

    # 2. Integrate Dynamic Features - From version 10 - second hp part
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
    lead_diff_features = _create_lead_differential_features(battles_df, teams_df)
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

    # 9. Boost Advantage Features
    print("Creating boost advantage features...")
    boost_features = _create_boost_advantage_features(turns_df)
    final_df = pd.merge(final_df, boost_features, on='battle_id', how='left')

    # 11. Big Three Features
    print("Creating big three features...")
    big_three_features = _create_big_three_features(teams_df, turns_df)
    final_df = pd.merge(final_df, big_three_features, on='battle_id', how='left')

    # 12. Trapping Moves Features 
    print("Creating recovery and trapping moves features...")
    trapping_moves_features = _create_trapping_features(turns_df)
    final_df = pd.merge(final_df, trapping_moves_features, on='battle_id', how='left')

    # 13. Recovery Moves Features 
    recovery_moves_features = _create_recovery_move_features(turns_df)
    final_df = pd.merge(final_df, recovery_moves_features, on='battle_id', how='left')

    # 14. Trading Moves Features
    recovery_moves_features = _create_trading_features(turns_df)
    final_df = pd.merge(final_df, recovery_moves_features, on='battle_id', how='left') 
    
    # 15. Strategic Switch Features
    print("Creating strategic switch features...")
    switch_features = _create_switch_features(turns_df)
    final_df = pd.merge(final_df, switch_features, on='battle_id', how='left')
    
    # 16. Predicted Damage Ratio on Turn 1
    print("Creating predicted damage ratio on turn 1 feature...")
    predicted_damage_t1_features = _create_predicted_damage_turn_1(battles_df, teams_df, turns_df)
    final_df = pd.merge(final_df, predicted_damage_t1_features, on='battle_id', how='left')

    # ------------------------- Final Cleanup --------------------------
    lead_types_col = ['p1_lead_types', 'p2_lead_types'] 
    p2_types_cols = [f'p2_lead_type_{t}' for t in POKEMON_TYPES]
    redundant_component_features = [
        'p1_lead_types', 
        'p2_lead_types',
        'p2_lead_name',
        'p2_lead_level',

            
        # 'lead_diff_spe',      #INF
        'lead_diff_hp',         # LIN TO lead_diff_hp => out for logistic 
        'lead_diff_def',      #INF
        'lead_diff_spa',      #INF
        'lead_diff_spd',      #INF
        'lead_diff_atk',      #INF
        # 'lead_diff_offense',   # INF  TEST
        # 'lead_diff_defense',   # INF   TEST
    
        'p1_def_core',      #INF
        'p2_def_core',  
        'p2_lead_base_atk',  # VERY HIGH
        'p2_lead_base_def',  # VERY HIGH
        'p2_lead_base_hp', 
        'p2_lead_base_spa',   # INF
        'p2_lead_base_spd',   # INF 
        'p2_lead_base_spe',   # INF
        
        
        'lead_spe_diff',       # INF   (Duplicate)
        'p1_team_mean_spa',    # INF
        'p1_team_mean_spd',    # INF
        'p1_team_mean_spe',    # INF  
        'p1_team_mean_atk',    # VERY HIGH
        # 'p1_team_mean_hp',     # VERY HIGH   
        'p1_team_mean_def',   # TEST TAKE OUT
        'lead_team_mean_spe_adv',    # INF
        #'p1_team_off_vs_p2_lead_def_ratio',  # VERY HIGH
        'type_matchup_diff',
        #'expected_damage_ratio_turn_1', 
        # 'p1_stab_count',
        # 'defensive_core_adv',     # INF   
        'p1_team_off_vs_p2_lead_def_ratio', 

        #'p2_hits_p1_not_effective',
        #'p2_hits_p1_super_effective',   #TEST  SEE IF THERE IS SUPER EFFECTIVE IN THERE
        'p1_switch_count',
        'p2_switch_count',  # TEST TAKEOUT FOR switch_advantage
        # 'switch_advantage',  # INF  #TEST

        'dynamic_first_move_ratio', 
        'first_move_advantage',   # OUT => lead_spd_diff is the equivalent 
          
        'defense_boost_advantage', 
        'offense_boost_advantage',
        'speed_boost_advantage',    
        
        'big_three_adv',    # INF  NOT NEEDED BECAUSE WE HAVE tauros and defensive core adv
        'chansey_adv',      # INF 
        'snorlax_adv',      # INF 
        #'tauros_adv',       # INF  TEST

        #'recovery_turn_adv',    # TEST
        #'trade_success_adv',   # TEST
        'trade_turn_adv', 
        #'trap_turn_advantage',   # TEST

        'p1_ko_count',   # INF  #TEST
        'p2_ko_count',
        #'ko_advantage',   # INF  #TEST  SEE IF KO FEATURE IS STILL CONTAINED
        #'p1_team_avg_hp',  # perfect collinearity  (we are using both averages and sum. I try with just the sum)
        #'p2_team_avg_hp',  #perfect collinearity 
        #'team_hp_advantage', #perfect collinearity 
    ]

    logistic_redundant_component_features = [

        'damage_pressure_advantage',
        'hp_trend_slope',
    ]
    
    
    # Define columns to drop: temp, original P2 types, and redundant components
    cols_to_drop = lead_types_col + p2_types_cols + redundant_component_features #+ logistic_redundant_component_features 
    final_df = final_df.drop(columns=cols_to_drop, errors='ignore').fillna(0)

    final_df = final_df.infer_objects(copy=False)
    print(f"Feature engineering version 13 completed. Feature count: {final_df.shape[1]}")
    return final_df