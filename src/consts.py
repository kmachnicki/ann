from os import path

BASE_DIR = path.dirname(path.dirname(__file__))
INPUT_DATA_FILE = path.join(BASE_DIR, "stopien_zlosliwosci.csv")
RANDOM_STATE = None