import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import numpy as np

# Importing project's modules
import src.data.load_data as load
import src.data.clean_data as clean
import src.data.feature_engineering.feature_engineering as feature_engineering

# --- Analysis Function 1: Correlation Heatmap ---
def plot_correlation_heatmap(X_df, save_path="correlation_heatmap.png"):
    """
    Calculates and plots a correlation heatmap for the feature DataFrame.
    Saves the plot to a file.
    """
    print("\n[Analysis] Calculating correlation matrix...")
    
    # Calculate the correlation matrix
    # Use 'spearman' for a more robust (non-linear) check, or 'pearson' for linear
    corr = X_df.corr(method='spearman')
    
    # Set up the matplotlib figure
    fig_width = max(15, len(X_df.columns) * 0.5) # Dynamic width
    fig_height = max(12, len(X_df.columns) * 0.4) # Dynamic height
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Draw the heatmap
    sns.heatmap(
        corr,
        mask=mask,         # Apply the mask
        ax=ax,
        cmap='vlag',       # Use a diverging colormap (blue-white-red)
        vmin=-1, vmax=1,   # Set the min/max for the colormap
        center=0,          # Center the colormap at 0
        annot=False,       # Do not show numbers (too many features)
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5}
    )
    
    ax.set_title("Feature Correlation Heatmap (Spearman)", fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300) # Save with high resolution
    print(f"[Analysis] Correlation heatmap saved to {save_path}")

# --- Analysis Function 2: Variance Inflation Factor (VIF) ---
def calculate_vif(X_df):
    """
    Calculates the VIF for each feature in the DataFrame.
    
    A VIF > 5 is a sign of multicollinearity.
    A VIF > 10 is a strong sign.
    """
    print("\n[Analysis] Calculating VIF scores...")
    
    # Create a copy to avoid modifying the original dataframe
    X_temp = X_df.copy()
    
    # VIF can't handle missing values, fill with 0 (or median/mean)
    X_temp = X_temp.fillna(0)
    
    # VIF calculation requires a constant (intercept)
    X_vals = add_constant(X_temp.values, prepend=True)

    # Create a DataFrame to store the VIF scores
    vif_data = pd.DataFrame()
    # Use the column names from the original DataFrame
    vif_data["feature"] = X_temp.columns
    
    # Calculate VIF for each feature
    # Note: We loop from 1 to skip the constant (at index 0)
    print("Calculating VIF for each feature. This may take a moment...")
    try:
        vif_data["VIF"] = [
            variance_inflation_factor(X_vals, i) 
            for i in range(1, X_vals.shape[1])
        ]
    except Exception as e:
        print(f"Error calculating VIF: {e}")
        print("This can be due to perfect collinearity or columns of all zeros.")
        return None

    # VIF will be 'inf' (infinite) for perfectly collinear features
    vif_sorted = vif_data.sort_values(by="VIF", ascending=False)
    
    print("\n--- Top 20 Highest VIF Scores ---")
    print(vif_sorted.head(20))
    
    # Check for VIFs of infinity
    inf_vif = vif_sorted[vif_sorted['VIF'] == np.inf]
    if not inf_vif.empty:
        print("\n[WARNING] Perfect Collinearity Detected (VIF = inf):")
        print(inf_vif)

    return vif_sorted


if __name__ == "__main__":
    print("\n ---------- Starting Feature Analysis ---------- \n")

    # 0. Config
    version = int(input("Select the feature engineering version to analyze:\n>>> ").strip())

    # ----------------------------------------------------------------------------------------
    # 1.1.  Load and clean train data (we only need train data for analysis)
    print("Loading training data...")
    raw_train_data = load.load_data(train=True)
    battles_train, turns_train, teams_train = clean.clean_data(raw_data=raw_train_data, train=True)

    # 1.2. Create features for train data
    print(f"Running feature engineering version {version}...")
    features_train = feature_engineering.feature_engineering(
        battles=battles_train,
        turns=turns_train,
        teams=teams_train,
        version=version,
        train=True)

    # 1.3. Prepare data for analysis
    # We want to check all features *before* dropping any
    # Renamed this variable from X_features to X_train to match your main.py
    print("Preparing feature DataFrame X_train...")
    X_train = features_train.drop(columns=["player_won", "battle_id"], errors='ignore')

    print(f"Analysis will be run on {len(X_train.columns)} features.")

    # ----------------------------------------------------------------------------------------
    # 2. Run Analyses

    # 2.1. Correlation Heatmap
    # Pass X_train to the function
    plot_correlation_heatmap(X_train, save_path=f"src/data/feature_engineering/analyze_features_outputs/feature_v{version}_correlation_heatmap.png")

    # 2.2. VIF Calculation
    # Pass X_train to the function
    vif_scores = calculate_vif(X_train)
    
    if vif_scores is not None:
        vif_scores.to_csv(f"src/data/feature_engineering/analyze_features_outputs/feature_v{version}_vif_scores.csv", index=False)
        print(f"\n[Analysis] VIF scores saved to feature_v{version}_vif_scores.csv")

    print("\n ---------- Analysis Complete ---------- \n")