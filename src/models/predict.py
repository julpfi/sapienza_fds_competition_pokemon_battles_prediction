import pandas as pd
import os
from datetime import datetime

def predict(model, data: pd.DataFrame, addition: str = ""): 

    test_predictions = model.predict(data)
    
    # Convert True/False to 1/0
    test_predictions = test_predictions.astype(int)

    submission_df = pd.DataFrame({
        'battle_id': data['battle_id'],
        'player_won': test_predictions
    })

    current_time = datetime.now()
    time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    name = time_string + (("_" + addition) if addition else "") + "_submission.csv"
    
    path = os.path.join("submissions", name)
    submission_df.to_csv(path, index=False)
    print(f"Saved to: {path}")