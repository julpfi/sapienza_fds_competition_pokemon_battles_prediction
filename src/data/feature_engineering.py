from data.clean_data import clean_data

def feature_engineering_version_1(): 
    return None


def feature_engineering(version: int=1):
    data = clean_data()
    match version: 
        case 1: 
            return feature_engineering_version_1
        case _: 
            raise Exception("ERROR: Invalid selection of which set of features to use. \n -> feature_engineering.py")