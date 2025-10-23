from src.data.load_data import load_data
from src.data.clean_data import clean_data
from src.data.feature_engineering import feature_engineering
from src.models.predict import predict

if __name__ == "__main__":
    
    version = int(input("Please select the feature engineering version:\n>>> "))

    raw_train_data = load_data(train=True)
    battles_train, turns_train, teams_train = clean_data(raw_data=raw_train_data, train=True)
    X_train, y_train = feature_engineering(version, train=True)


    model = None 

    raw_test_data = load_data(train=True)
    battles_test, turns_test, teams_test = clean_data(raw_data=raw_test_data, train=True)
    X_test = feature_engineering(version, train=True)

    predict(model, X_test)