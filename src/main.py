import src.data.load_data as load
import src.data.clean_data as clean
import src.data.feature_engineering.feature_engineering as feature_engineering
import src.models.predict  as predict
import src.models.train as train

if __name__ == "__main__":
    print("\n ---------- Starting ML Pipeline ---------- \n")

    # Config
    version = int(input("Select the feature engineering version:\n>>> ").strip())
    model_map = {
        1 : "logistic",
        2: "random_forest",
        3 : "xgboost",
        4 : "voting"
    }
    model_type = model_map.get(int(input("Select model to use: \n    1 - Logistic regression\n    2 - Random forest\n    3 - XGBoost\n    4 - Voting (1-2)\n>>> ").strip()))
    with_grid_search = input("With GridSearch (y/n)\n>>> ").strip().lower() == 'y'
    print("\n")
    
    # ----------------------------------------------------------------------------------------
    # 1.1.  Load and clean train data
    raw_train_data = load.load_data(train=True)
    battles_train, turns_train, teams_train = clean.clean_data(raw_data=raw_train_data, train=True)

    # 1.2. Create features for train data
    features_train = feature_engineering.feature_engineering(
        battles=battles_train,
        turns=turns_train,
        teams=teams_train,
        version=version,
        train=True)

    y_train = features_train["player_won"]

    X_train = features_train.drop(columns=["player_won", "battle_id"])

    # 1.3. Train model including GridSearch (for LR data will be standardized)  
    model = train.train(X=X_train, y=y_train, model_type=model_type, grid_search=with_grid_search)


    # ----------------------------------------------------------------------------------------
    predict_and_create_csv = input("Predict test data and create csv (y/n)\n>>> ").lower() == 'y'

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
            train=False)

        battle_ids_test = features_test["battle_id"]
        X_test = features_test.drop(columns=['battle_id', 'player_won'])   
         
    # 2.3. Predict test data with model and save file (for LR data will be standardized)
        addition = input("Add text to submission.csv file name\n>>> ")
        predict.predict(model, X_test, battle_ids=battle_ids_test, addition=addition)
