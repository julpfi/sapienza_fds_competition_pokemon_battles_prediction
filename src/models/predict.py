import pandas as pd
import os
from utils.config import SUBMISSION_DIR
from datetime import datetime

def predict(model, data:pd.DataFrame): 

    test_predictions = model.predict(model)

    submission_df = pd.DataFrame({
        'battle_id': data['battle_id'],
        'player_won': test_predictions
    })

    current_time = datetime.now()
    time_string = current_time.strftime("%m/%d/%Y, %H:%M:%S")
    name = time_string + 'submission.csv'
    path = os.path.join(path,name)
    submission_df.to_csv('submission.csv', index=False)



