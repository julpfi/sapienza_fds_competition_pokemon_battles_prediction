import pandas as pd
import numpy as np


def remove_low_variance_features(df: pd.DataFrame, threshold: float = 0.01, exclude_cols: list = None) -> pd.DataFrame:
    if exclude_cols is None:
        exclude_cols = []
    
    variances = df.drop(columns=exclude_cols, errors='ignore').var()
    low_var_cols = variances[variances < threshold].index.tolist()
    
    if low_var_cols:
        print(f"Removing {len(low_var_cols)} low-variance features")
        df = df.drop(columns=low_var_cols)
    
    return df


def remove_highly_correlated_features(df: pd.DataFrame, threshold: float = 0.95, exclude_cols: list = None) -> pd.DataFrame:
    if exclude_cols is None:
        exclude_cols = []
    
    # Safety check: only consider numeric columns for correlation (should already be the case)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_check = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(cols_to_check) < 2:
        return df
    
    corr_matrix = df[cols_to_check].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    if to_drop:
        print(f"Removing {len(to_drop)} highly correlated features")
        df = df.drop(columns=to_drop)
    
    return df


def select_features(df: pd.DataFrame, variance_threshold: float = 0.01, 
                    correlation_threshold: float = 0.95, exclude_cols: list = None) -> pd.DataFrame:

    if exclude_cols is None:
        exclude_cols = ['battle_id', 'player_won']
    
    print(f"\nFilter-based feature selection:")
    number_columns_before = df.shape[1] - len([c for c in exclude_cols if c in df.columns])
    print(f"Initial features: {number_columns_before}")

    # Drop non-numeric columns except those in exclude_cols
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numeric_cols = [col for col in non_numeric_cols if col not in exclude_cols]
    if non_numeric_cols:
        print(f"Dropping {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols}")
        df = df.drop(columns=non_numeric_cols, errors='ignore')

    # Remove low variance and highly correlated features
    df = remove_low_variance_features(df, threshold=variance_threshold, exclude_cols=exclude_cols)
    df = remove_highly_correlated_features(df, threshold=correlation_threshold, exclude_cols=exclude_cols)

    # Infer datatypes before returning (ensures float/int consistency)
    df = df.infer_objects(copy=False)

    number_columns_after = df.shape[1] - len([c for c in exclude_cols if c in df.columns])
    print(f"Final features: {number_columns_after}\n")
    
    return df
