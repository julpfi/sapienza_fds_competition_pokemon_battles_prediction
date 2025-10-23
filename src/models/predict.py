import pandas as pd

def predict(model, data:pd.DataFrame): 
    print("Generating predictions on the test set...")
    test_predictions = model.predict(X_test)

    # Create the submission DataFrame
    submission_df = pd.DataFrame({
        'battle_id': test_df['battle_id'],
        'player_won': test_predictions
    })

    # Save the DataFrame to a .csv file
    submission_df.to_csv('submission.csv', index=False)

    print("\n'submission.csv' file created successfully!")
    display(submission_df.head())




    load_data()


