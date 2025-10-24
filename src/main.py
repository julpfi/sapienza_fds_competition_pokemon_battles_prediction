import src.data.load_data as load
import src.data.clean_data as clean
import src.data.feature_engineering.feature_engineering as feature_engineering
import src.models.predict  as predict
import src.models.train as train
import pandas

if __name__ == "__main__":
    
    version = int(input("Please select the feature engineering version:\n>>> "))

    raw_train_data = load.load_data(train=True)
    battles_train, turns_train, teams_train = clean.clean_data(raw_data=raw_train_data, train=True)

    X_train, y_train = feature_engineering.feature_engineering(
        battles=battles_train,
        turns=turns_train,
        teams=teams_train,
        version=version,
        train=True)

    model = train.train_baseline(X=X_train, y=y_train)

    raw_test_data = load.load_data(train=False)
    battles_test, turns_test, teams_test = clean.clean_data(raw_data=raw_test_data, train=False)
    
    X_test = feature_engineering.feature_engineering(
        battles=battles_test, 
        turns=turns_test, 
        teams=teams_test, 
        version=version, 
        train=False)


    #predict.predict(model, X_test,
    #                input("Add text to submission.csv file name\n>>> "))
                    