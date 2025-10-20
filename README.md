# sapienza_fds_competition_pokemon_battles_prediction
## Team: Julius Pfingsten, Ludovico Piras, and Nicolò Boscherini

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

**1. Install the package**
```bash
conda install -c conda-forge <package-name>
```

**2. Manually add to `environment.yml`**
- Open `environment.yml` and add the package alphabetically in the dependencies list
- Format: `- <package-name>=<version>`
- Example: If you installed `pandas=2.3.3`, add it between `optuna` and `pandocfilters`

**3. Commit changes**
```bash
git add environment.yml
git commit -m "Add <package-name> dependency"
git push
```

### For Pip packages

**1. Install the package**
```bash
pip install <package-name>
```

**2. Manually add to `environment.yml`**
- Open `environment.yml` and add under the `- pip:` section (before `- -e .`)
- Format: `- <package-name>==<version>`

**3. Commit changes**
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

**If issues occur, recreate the environment:**
```bash
git pull
conda deactivate
conda env remove -n sapienza_fds_pokemon
conda env create -f environment.yml
conda activate sapienza_fds_pokemon
```
