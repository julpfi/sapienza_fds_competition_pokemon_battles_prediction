
MODEL_MAP_MAIN = {
        1 : "logistic",
        2 : "random_forest",
        3 : "xgboost",
        4 : "knn",
        5 : "hgb",
        6 : "custom_voting",
        7 : "voting"
    }

MODEL_MAP_VOTING = {
        1 : "logistic",
        2 : "random_forest",
        3 : "xgboost",
        4 : "knn",
        5 : "hgb"
    }

# Like voting but without xgboost
MODEL_MAP_CUSTOM_VOTING = {
        1 : "logistic",
        2 : "random_forest",
        3 : "knn",
        4 : "hgb"
    }

def _print_selection(mapping:dict):
    s = ""
    s = s + "Select model to use:"
    for key in mapping:
        s = s + "    " + key + " - " + mapping[key] + "\n"
    
    s = s + ">>> "
    return s 

def get_user_model_selection_main():
    map = int(input(_print_selection(MODEL_MAP_MAIN)).strip())
    model_type = MODEL_MAP_MAIN.get(map)
    return model_type


def get_user_model_selection_voting():
    answer = input(_print_selection(MODEL_MAP_VOTING)).strip()
    if answer == '':
        return [k for k in MODEL_MAP_VOTING.values()]
    else: 
        selections = [int(x) for x in answer.split(',')]
        model_types = [MODEL_MAP_VOTING.get(x) for x in selections]
        return model_types


def get_user_model_selection_custom_voting():
    answer = input(_print_selection(MODEL_MAP_CUSTOM_VOTING)).strip()
    if answer == '':
        return [k for k in MODEL_MAP_CUSTOM_VOTING.values()]
    else: 
        selections = [int(x) for x in answer.split(',')]
        model_types = [MODEL_MAP_CUSTOM_VOTING.get(x) for x in selections]
        return model_types

