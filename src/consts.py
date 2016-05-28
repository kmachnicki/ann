from os import path

BASE_DIR = path.dirname(path.dirname(__file__))
INPUT_DATA_FILE = path.join(BASE_DIR, "stopien_zlosliwosci.csv")
OUTPUT_IMAGES_DIR = path.join(BASE_DIR, "results")
RANDOM_STATE = None
HIDDEN_LAYER_SIZES = [5, 10, 20, 40, 60]
N_RUNS = 2
K_BEST_FEATURES = 5
N_FOLDS = 10
BP_ALGORITHM = "sgd"
BP_MAX_ITER = 10000
BP_ALPHA = 1e-6
BP_LEARNING_RATE = "constant"
BP_ACTIVATION_FUNC = "logistic"
ELM_ACTIVATION_FUNC = "multiquadric"
