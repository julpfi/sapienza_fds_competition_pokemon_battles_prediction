
# paths 
DATA_RAW_DIR = "data/raw/"
DATA_INTERIM_DIR = "data/interim/"
DATA_CLEANED_DIR = "data/cleaned/"

DATA_TRAIN_PATH = f"{DATA_RAW_DIR}train.jsonl"
DATA_TEST_PATH = f"{DATA_RAW_DIR}test.jsonl"
MODEL_DIR = "models/"
SUBMISSION_DIR = "submissions/"

# seed
SEED = 1019

# constants: 
# Need to check if attacks have more types -> 
POKEMON_TYPES = ['water', 'grass', 'fire', 'psychic', 'rock', 'ice', 'ground', 'flying', 'normal', 'dragon', 'notype', 'electric', 'poison', 'ghost']
MOVE_CATEGORIES = ['SPECIAL', 'PHYSICAL', 'STATUS']