#Quick EDA 
#Just checking for null values, outliers and basic stats about our features

# --- BATTLES_DF ANALYSIS (P2 Lead Stats) ---
print("="*50)
print("BATTLES_DF ANALYSIS:")
print("Missing Values (Nulls):")
print(train_battles_df.isnull().sum())
print("\nData Types (Info):")
train_battles_df.info(verbose=False, memory_usage=False)
print("\nDescriptive Stats:")
print(train_battles_df.describe())
print("="*50)

# --- TEAMS_DF ANALYSIS (P1 Team Stats - 6 rows per battle) ---
print("\nTEAMS_DF ANALYSIS:")
print("Missing Values (Nulls):")
print(train_teams_df.isnull().sum())
print("\nData Types (Info):")
train_teams_df.info(verbose=False, memory_usage=False)
print("\nDescriptive Stats:")
print(train_teams_df.describe())
print("="*50)

# --- TURNS_DF ANALYSIS (Timeline Dynamics) ---
print("\nTURNS_DF ANALYSIS:")
print("Missing Values (Nulls):")
print(train_turns_df.isnull().sum())
print("\nData Types (Info):")
train_turns_df.info(verbose=False, memory_usage=False)
print("\nHP Descriptive Stats:")
# Just focusing on the most important columns for a quick check
print(train_turns_df[['p1_pokemon_state_hp_pct', 'p2_pokemon_state_hp_pct']].describe())

# Check all unique status values:
print("\nP1 Status Counts:")
print(train_turns_df['p1_pokemon_state_status'].value_counts(dropna=False))

print("\nP2 Status Counts:")
print(train_turns_df['p2_pokemon_state_status'].value_counts(dropna=False))
print("="*50)