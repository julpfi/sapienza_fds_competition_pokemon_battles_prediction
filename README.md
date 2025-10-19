# sapienza_fds_competition_pokemon_battles_prediction
## Team: Julius Pfingsten, Ludovido Piras, and Nicolo Boscherini


## Setting up anaconda environment
1. Create environment
conda env create -n sapienza_fds_pokemon -f environment.yml

2. Activate environment
conda activate sapienza_fds_pokemon

## Adding library to anaconda environment
1. Install
conda install -c conda-forge <package-name>

2. Update environment.yml
conda env export --from-history > environment.yml

3. Manual cleanup (important!)
Open environment.yml and:
 - Remove "prefix:" line at the end
 - Ensure pip section is still there with "-e ."

4. Commit
git add environment.yml
git commit -m "Add <package-name> dependency"
git push

## Taking over anaconda environment changes from others
git pull
conda activate sapienza_fds_pokemon
conda env update -f environment.yml --prune