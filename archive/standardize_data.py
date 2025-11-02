
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List

_GLOBAL_SCALER = StandardScaler()

def standardize_data(data: pd.DataFrame, train: bool = True, features_to_exclude: list[str] = ['battle_id', 'player_won']) -> pd.DataFrame:
    """
    Description: 
        Standardizes numerical features using StandardScaler. It fits the scaler ONLY if train=True.
    Params:
        data (pd.DataFrame): The DataFrame (train or test) containing features
        train (bool): If True, fits the scaler and returns X_scaled and y
                      If False, only transforms data and returns X_scaled
        features_to_exclude (list[str]): Columns to ignore during scaling
        
    Returns:
        pd.DataFrame or Tuple[pd.DataFrame, pd.Series]: Scaled features (X) and optionally the target (y).
    """
    m = "train" if train else "test"
    print(f"Start standardization of {m} data")
    
    # 1. Feature and Target Definition
    
    # Robustly select only NUMERICAL features and create a copy.
    X = data.select_dtypes(include=np.number).copy()
    
    # Separate the target (y) and remove identifying/target columns from X.
    if train:
        # Target must be present only in the training set
        y = data['player_won'].astype(int) 
        
    # Remove identifying columns from feature set X
    X = X.drop(columns=features_to_exclude, errors='ignore')
    
    # Preserve feature names for the scaled DataFrame
    features_list = X.columns.tolist()

    # 2. Fit or Transform
    
    global _GLOBAL_SCALER # Access the scaler defined outside the function
    
    if train:
        # CRITICAL: We FIT (learn mean/std) and TRANSFORM ONLY on the Training set.
        print("Fitting scaler and transforming training data...")
        X_scaled = _GLOBAL_SCALER.fit_transform(X) 
    else:
        # CRITICAL: We ONLY TRANSFORM using the parameters learned during the fit phase.
        print("Transforming test/new data using fitted scaler...")
        X_scaled = _GLOBAL_SCALER.transform(X) 
   
    return pd.DataFrame(X_scaled, columns=features_list)