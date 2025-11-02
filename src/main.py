import src.data.load_data as load
import src.data.clean_data as clean
import src.data.feature_engineering.feature_engineering as feature_engineering
import src.models.predict  as predict
import src.models.train as train

if __name__ == "__main__":
    
    version = int(input("Please select the feature engineering version:\n>>> "))
    with_grid_search = input("With GridSearch (y/n)\n>>> ").lower() == 'y'
    model_type = "logistic"


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
    X_train = features_train.drop(columns=["player_won"])

    # 1.3. Train model including GridSearch (for LR data will be standardized)  
    model = train.train(X=X_train, y=y_train, model_type=model_type, grid_search=with_grid_search)


    # ----------------------------------------------------------------------------------------
    predict_and_create_csv = input("Predict test data and create csv (y/n)\n>>> ").lower() == 'y'
    
    if predict_and_create_csv: 
    # 2.1. Load and clean test data
        raw_test_data = load.load_data(train=False)
        battles_test, turns_test, teams_test = clean.clean_data(raw_data=raw_test_data, train=False)
    
    # 2.2. Create features for test data
        X_test = feature_engineering.feature_engineering(
            battles=battles_test, 
            turns=turns_test, 
            teams=teams_test, 
            version=version, 
            train=False)

    # 2.3. Predict test data with model and save file (for LR data will be standardized)
        addition = input("Add text to submission.csv file name\n>>> ")
        predict.predict(model, X_test, addition=addition, model_type=model_type)
