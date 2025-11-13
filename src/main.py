import src.data.load_data as load
import src.data.clean_data as clean
import src.data.feature_engineering.feature_engineering as feature_engineering
import src.models.predict  as predict
import src.models.train as train
from src.utils.user_model_selection import get_user_model_selection_main
from data.feature_engineering.features_version_14 import _create_pokemon_stats_map

if __name__ == "__main__":
    print("\n ---------- Starting ML Pipeline ---------- \n")

    # 0. Config
    version = int(input("Select the feature engineering version:\n>>> ").strip())
    model_type = get_user_model_selection_main()

    with_grid_search = True #input("With GridSearch (y/n)\n>>> ").strip().lower() == 'y'
    print("Selecting model:", model_type, "\nUsing GridSeach (RandomSearch for XGB and HGB)")
    print("\n")
    
    # ----------------------------------------------------------------------------------------
    # 1.1.  Load and clean train data
    raw_train_data = load.load_data(train=True)
    battles_train, turns_train, teams_train = clean.clean_data(raw_data=raw_train_data, train=True)
    pokedex = None
    default_stats = None
    if version == 14:
        pokedex, default_stats = _create_pokemon_stats_map(teams_df=teams_train, battles_df=battles_train)

    # 1.2. Create features for train data
    features_train = feature_engineering.feature_engineering(
        battles=battles_train,
        turns=turns_train,
        teams=teams_train,
        version=version,
        train=True,
        pokemon_stats_map=pokedex,
        default_pokemon_stats=default_stats)

    y_train = features_train["player_won"]

    X_train = features_train.drop(columns=["player_won", "battle_id"])

    # 1.3. Train model including GridSearch (for LR data will be standardized)  
    model = train.train(X=X_train, y=y_train, model_type=model_type, grid_search=with_grid_search)


    # ----------------------------------------------------------------------------------------
    predict_and_create_csv = input("\nPredict test data and create csv (y/n)\n>>> ").strip().lower() == 'y'

    if predict_and_create_csv: 
    # 2.1. Load and clean test data
        raw_test_data = load.load_data(train=False)
        battles_test, turns_test, teams_test = clean.clean_data(raw_data=raw_test_data, train=False)
    
    # 2.2. Create features for test data
        features_test = feature_engineering.feature_engineering(
            battles=battles_test, 
            turns=turns_test, 
            teams=teams_test, 
            version=version, 
            train=False,
            pokemon_stats_map=pokedex,
            default_pokemon_stats=default_stats)

        battle_ids_test = features_test["battle_id"]
        X_test = features_test.drop(columns=['battle_id', 'player_won'])   
         
    # 2.3. Predict test data with model and save file (for LR data will be standardized)
        addition = input("Add text to submission.csv file name\n>>> ")
        predict.predict(model, X_test, battle_ids=battle_ids_test, addition=addition)
