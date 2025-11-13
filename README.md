# Kaggle Challenge - Pokemon Battles Prediction 2025 

## Information
1. Course: Fundamentals of Data Science (9 CFU)
2. Team: Julius Pfingsten, Ludovico Piras, and Nicolò Boscherini
3. Repo Name: sapienza_fds_competition_pokemon_battles_prediction


## Workflow
The `main` method in the `src` folder functions as the entrypoint for running the whole predicting pipeline. It calls the data pipeline (including loading, cleaning, and extracting), the feature engineering, the model training (including tuning and validating the model via GridSearch/RandomSearch), and if selected, the prediction step which also saves the submission. We added terminal-based user selection for an efficient execution. 

Additionally, we use the `analyze_feature` method in `src/data/feature_engineering` to create the vif score and correlation heatmap. Any other exploration and testing was done in the `notebooks` folder or in Kaggle notebooks. 


## Accessing the Correct Submission
During our work, the whole pipeline was improved iteratively on various points. To ensure that the accurate version for one submission can be found later on, we established the following workflow:

1.  Every time before predicting a new submission, we push the whole repository and ensure it is at the head of the `main` branch.
2.  Then, with the prediction, we add the new file to the `submission` folder.
3.  That change is then pushed with the commit message "SUBMISSION [name of submission]" to ensure it can be found in the commit history.

Although this setup might not be perfect, it worked well as a simple workaround to connect repo versions with submission files.


## Setting up Anaconda Environment

### 1. Create environment
```bash
conda env create -f environment.yml
```

### 2. Activate environment
```bash
conda activate sapienza_fds_pokemon
```

---

## Adding Packages to Anaconda Environment

### For Conda packages

1. Install the package
```bash
conda install -c conda-forge <package-name>
```

2. Manually add to `environment.yml`
- Open `environment.yml` and add the package alphabetically in the dependencies list
- Format: `- <package-name>=<version>`
- Example: If you installed `pandas=2.3.3`, add it between `optuna` and `pandocfilters`

3. Commit changes
```bash
git add environment.yml
git commit -m "Add <package-name> dependency"
git push
```

### For Pip packages

1. Install the package
```bash
pip install <package-name>
```

2. Manually add to `environment.yml`
- Open `environment.yml` and add under the `- pip:` section (before `- -e .`)
- Format: `- <package-name>==<version>`

3. Commit changes
```bash
git add environment.yml
git commit -m "Add <package-name> dependency"
git push
```

---

## Taking Over Environment Changes from Others
```bash
git pull
conda activate sapienza_fds_pokemon
conda env update -f environment.yml --prune
```

If issues occur, recreate the environment:
```bash
git pull
conda deactivate
conda env remove -n sapienza_fds_pokemon
conda env create -f environment.yml
conda activate sapienza_fds_pokemon
```
