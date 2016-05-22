from os import path

BASE_DIR = path.dirname(path.dirname(__file__))
INPUT_DATA_FILE = path.join(BASE_DIR, "stopien_zlosliwosci.csv")
OUTPUT_IMAGES_DIR = path.join(BASE_DIR, "results")
RANDOM_STATE = None
HIDDEN_LAYER_SIZES = [100, 250, 500, 750, 1000]
N_RUNS = 5
K_BEST_FEATURES = 5
N_FOLDS = 10
BP_ALGORITHM = "sgd"
BP_MAX_ITER = 1000
BP_ALPHA = 1e-6
BP_LEARNING_RATE = "constant"
ELM_ACTIVATION_FUNC = "multiquadric"
