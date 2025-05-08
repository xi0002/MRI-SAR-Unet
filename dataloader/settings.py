PRINT_MODEL = False
FILTERS = 16
USE_UPSAMPLING = False

BATCH_SIZE = 4
TILE_HEIGHT = 144
TILE_WIDTH = 144
TILE_DEPTH = 144
NUMBER_INPUT_CHANNELS = 5
NUMBER_OUTPUT_CLASSES = 3

TRAIN_TEST_SPLIT = 0.825  # 0.9
VALIDATE_TEST_SPLIT = 0.50
FILE_RATIO = 1

# 1e-3 for SSol
initial_lr = 1e-3

RANDOM_SEED = 830  # 826
SAVED_MODEL_NAME = "saved_model"
EPOCHS = 100
# input
hard_normalizing_input = True
input_with_E_inc = True
input_with_E_angle = False
shuffle_during_training = True
# output
X_DATA_PATH = "./data/X_DATA_PATH/"
Y_DATA_PATH = "./data/Y_DATA_PATH/"

# model
out_lay_act = "none"
