import pandas as pd
import os
from datetime import datetime
from data.standardize_data import standardize_data

def predict(model, data: pd.DataFrame, addition: str = "", model_type:str="logistic"): 
    '''
    Description:  
        Predicts the outcome of the test data with the given model. 
        Creates and saves the submissions csv to our submissions folder. 
        NOTE: To track when what submission was trained, commit and push repo after training with submission in commit message
    Param: 
        mode: Model that is used to predict data
        data (pd.DataFrame): test data that was prepared the same way as the train data. 
        addition (str): Any additional naming for the title of the csv that will be saved
        model_type (str): model_type which is relevant for standardization of data needed for certain models
    '''
    print("Starting prediction of test data")
    
    # Standardize data for logistic regression 
    if model_type == "logistic": 
        data = standardize_data(data, train=False)

    # Predict test data 
    test_predictions = model.predict(data)
    
    # Convert True/False to 1/0
    test_predictions = test_predictions.astype(int)

    # Create df in submission format 
    submission_df = pd.DataFrame({
        'battle_id': data['battle_id'],
        'player_won': test_predictions
    })

    # Add time and addition to submissions naming
    current_time = datetime.now()
    time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    name = time_string + (("_" + addition) if addition else "") + "_submission.csv"
    
    # Save submission csv
    path = os.path.join("submissions", name)
    submission_df.to_csv(path, index=False)
    print(f"Saved to: {path}")