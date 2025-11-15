
#This script is a standalone tool for analyzing the feature engineering versions.
#It performs:
#1. Correlation Heatmap (Spearman)
#2. Variance Inflation Factor (VIF) Analysis (to detect collinearity)
#3. AUC/ROC Plot (based on XGBoost performance)

#=============================================================================
 #IMPORTANT                                       
# This script is designed to work with feature engineering versions 6-13.  
#                                                                        
# It will NOT run with version 14                             
#                                                                        


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import numpy as np
import sys # <-- ADDED THIS IMPORT

# --- New Imports for Model Training & AUC Plot ---
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
# -------------------------------------------------

# --- NEW Imports for Meta-Model ---
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# ----------------------------------

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
        mask=mask,
        ax=ax,
        cmap='vlag',
        vmin=-1, vmax=1,
        center=0,
        annot=False,
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
    """
    print("\n[Analysis] Calculating VIF scores...")
    
    X_temp = X_df.copy().fillna(0)
    X_vals = add_constant(X_temp.values, prepend=True)

    vif_data = pd.DataFrame()
    vif_data["feature"] = X_temp.columns
    
    print("Calculating VIF for each feature. This may take a moment...")
    try:
        vif_data["VIF"] = [
            variance_inflation_factor(X_vals, i) 
            for i in range(1, X_vals.shape[1])
        ]
    except Exception as e:
        print(f"Error calculating VIF: {e}")
        return None

    vif_sorted = vif_data.sort_values(by="VIF", ascending=False)
    
    print("\n--- Top 20 Highest VIF Scores ---")
    print(vif_sorted.head(20))
    
    inf_vif = vif_sorted[vif_sorted['VIF'] == np.inf]
    if not inf_vif.empty:
        print("\n[WARNING] Perfect Collinearity Detected (VIF = inf):")
        print(inf_vif)

    return vif_sorted

# --- Analysis Function 3: Simple XGBoost AUC/ROC Plot ---
def plot_auc_roc_curve(X, y, save_path="auc_roc_plot.png"):
    """
    Trains a simple XGBoost model to generate and save an AUC/ROC plot.
    Uses a train/test split to get a realistic score.
    """
    print("\n[Analysis] Generating AUC/ROC Plot...")
    
    # 1. Split data into temporary train/validation sets
    # We use this split so the model doesn't predict on data it already saw
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.3,    # 30% for validation
        random_state=42,  # for reproducibility
        stratify=y        # Ensures win/loss ratio is same in both sets
    )
    
    # 2. Train a basic XGBoost model
    print("Training temporary XGBoost model for AUC...")
    model = xgb.XGBClassifier(
        random_state=42, 
        eval_metric='logloss',
        use_label_encoder=False # Suppress warning
    )
    model.fit(X_train, y_train)
    
    # 3. Get predicted probabilities for the validation set
    # We need the probability of the "positive" class (player_won=1)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # 4. Calculate ROC curve data
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # 5. Plot
    plt.figure(figsize=(10, 8))
    plt.plot(
        fpr, 
        tpr, 
        color='darkorange', 
        lw=2, 
        label=f'XGBoost ROC curve (AUC = {roc_auc:.3f})'
    )
    plt.plot(
        [0, 1], 
        [0, 1], 
        color='navy', 
        lw=2, 
        linestyle='--',
        label='Random Guess'
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    print(f"[Analysis] AUC/ROC plot saved to {save_path}")
# --- End of function ---


# --- NEW Analysis Function 4: Meta-Model AUC/ROC Plot ---
def plot_meta_model_auc_roc_curve(X, y, save_path="meta_auc_roc_plot.png"):
    """
    Trains a StackingClassifier (LR+KNN+XGB) to generate and save an
    AUC/ROC plot. Uses default parameters for the base models.
    """
    print("\n[Analysis] Generating Meta-Model AUC/ROC Plot...")
    
    # 1. Split data into temporary train/validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.3,    # 30% for validation
        random_state=42,  # for reproducibility
        stratify=y        # Ensures win/loss ratio is same in both sets
    )
    
    # 2. Define Base Estimators (using defaults, as we can't grid search here)
    # We must scale data for LR and KNN
    
    # LR Pipeline
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'))
    ])
    
    # KNN Pipeline
    knn_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier()) # n_neighbors=5 (default)
    ])

    # XGB Model (no scaling needed)
    xgb_model = xgb.XGBClassifier(
        random_state=42, 
        eval_metric='logloss',
        use_label_encoder=False
    )

    estimators = [
        ('lr', lr_pipe),
        ('knn', knn_pipe),
        ('xgb', xgb_model)
    ]
    
    # 3. Define the Stacking Classifier
    # This is the 'meta_model'
    meta_estimator = LogisticRegression(max_iter=1000)
    
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_estimator,
        cv=5, # Use 5-fold CV on the training data, as in your code
        n_jobs=-1
    )
    
    # 4. Train the Stacker
    print("Training temporary StackingClassifier for AUC...")
    stacking_model.fit(X_train, y_train)
    
    # 5. Get probabilities for the validation set
    y_pred_proba = stacking_model.predict_proba(X_val)[:, 1]
    
    # 6. Calculate ROC curve data
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # 7. Plot
    plt.figure(figsize=(10, 8))
    plt.plot(
        fpr, 
        tpr, 
        color='darkorange', 
        lw=2, 
        label=f'Meta-Model ROC curve (AUC = {roc_auc:.3f})'
    )
    plt.plot(
        [0, 1], 
        [0, 1], 
        color='navy', 
        lw=2, 
        linestyle='--',
        label='Random Guess'
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Meta-Model Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    print(f"[Analysis] Meta-Model AUC/ROC plot saved to {save_path}")
# --- End of new function ---


if __name__ == "__main__":
    print("\n ---------- Starting Feature Analysis ---------- \n")

    # 0. Config
    version = int(input("Select the feature engineering version to analyze:\n>>> ").strip())

    # --- NEW: Add check for version 14+ ---
    if version >= 14 or version == 5:
        print("\n[ANALYSIS NOTE]")
        print("="*60)
        print(f"You selected Version {version}.")
        print("This standalone analysis script is designed for versions 6-13.")
        print("="*60)
        sys.exit() # Exit the script gracefully
    # --- END NEW SECTION ---

    # ----------------------------------------------------------------------------------------
    # 1.1.  Load and clean train data
    print("Loading training data...")
    raw_train_data = load.load_data(train=True)
    battles_train, turns_train, teams_train = clean.clean_data(raw_data=raw_train_data, train=True)

    # 1.2. Create features for train data
    print(f"Running feature engineering version {version}...")
    
    # --- FIX: Reverted to the simpler feature engineering call ---
    # The v14+ logic was buggy and has been replaced by the
    # check above.
    features_train = feature_engineering.feature_engineering(
        battles=battles_train,
        turns=turns_train,
        teams=teams_train,
        version=version,
        train=True
        )

    # 1.3. Prepare data for analysis
    print("Preparing feature DataFrames X and y...")
    # --- ADDED: Create y_train ---
    y_train = features_train["player_won"]
    X_train = features_train.drop(columns=["player_won", "battle_id"], errors='ignore')

    print(f"Analysis will be run on {len(X_train.columns)} features.")

    # ----------------------------------------------------------------------------------------
    # 2. Run Analyses

    # 2.1. Correlation Heatmap
    plot_correlation_heatmap(X_train, save_path=f"src/data/feature_engineering/analyze_features_outputs/feature_v{version}_correlation_heatmap.png")

    # 2.2. VIF Calculation
    vif_scores = calculate_vif(X_train)
    
    # --- 2.3. Create a clean DataFrame for modeling ---
    # We train the model on a "clean" set of features for a more realistic plot
    # These are the components that cause 'inf' VIF scores
    features_to_drop_for_model = [
        'p1_ko_count', 'p2_ko_count',
        'p1_pokemon_left', 'p2_pokemon_left',
        'p1_team_avg_hp', 'p2_team_avg_hp',
        'p1_team_sum_hp', 'p2_team_sum_hp'
    ]
    
    # Create a clean DataFrame for the model
    X_train_clean_for_model = X_train.drop(columns=features_to_drop_for_model, errors='ignore')
    
    print(f"\n[Analysis] Using {len(X_train_clean_for_model.columns)} 'clean' features for AUC plot models.")
    
    # --- 2.4. Plot Simple XGBoost AUC (for feature analysis) ---
    plot_auc_roc_curve(
        X_train_clean_for_model, 
        y_train, 
        save_path=f"src/data/feature_engineering/analyze_features_outputs/feature_v{version}_XGB_auc_roc_plot.png"
    )
    
    # --- 2.5. NEW: Plot Meta-Model AUC (for final report) ---
    plot_meta_model_auc_roc_curve(
        X_train_clean_for_model,
        y_train,
        save_path=f"src/data/feature_engineering/analyze_features_outputs/feature_v{version}_META_auc_roc_plot.png"
    )
    # --- End of new section ---
    
    if vif_scores is not None:
        vif_scores.to_csv(f"src/data/feature_engineering/analyze_features_outputs/feature_v{version}_vif_scores.csv", index=False)
        print(f"\n[Analysis] VIF scores saved to feature_v{version}_vif_scores.csv")

    print("\n ---------- Analysis Complete ---------- \n")