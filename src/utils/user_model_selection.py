
MODEL_MAP_MAIN = {
        1 : "logistic",
        2 : "random_forest",
        3 : "xgboost",
        4 : "knn",
        5 : "hgb",
        6 : "custom_voting",
        7 : "voting",
        8 : "meta"
    }

MODEL_MAP_BASE = {
        1 : "logistic",
        2 : "random_forest",
        3 : "knn",
        4 : "hgb",
        5: "xgboost"
    }


def _print_selection(mapping:dict, addition:str="")-> str:
    s = ""
    s = s + "Select model to use:\n"
    for key in mapping:
        s = s + "    " + str(key) + " - " + mapping[key] + "\n"
    
    s = s + addition
    s = s + ">>> "
    return s 


def get_user_model_selection_main():
    map = int(input(_print_selection(MODEL_MAP_MAIN)).strip())
    model_type = MODEL_MAP_MAIN.get(map)
    return model_type


def get_user_model_selection(include_xgb:bool=True):
    map = {}
    if include_xgb:
        map = MODEL_MAP_BASE
    else: 
        map = {k:v for k,v in MODEL_MAP_BASE.items() if v != 'xgboost'}

    answer = input(_print_selection(map, addition="Selecting number seperated by a comma\nFor all: press enter\n")).strip()

    if answer == '':
        return [k for k in map.values()]
    else: 
        selections = [int(x) for x in answer.split(',')]
        model_types = [map.get(x) for x in selections]
        return model_types
    