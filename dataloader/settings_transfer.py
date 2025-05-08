PRINT_MODEL = False
FILTERS = 16
USE_UPSAMPLING = False

BATCH_SIZE = 4
TILE_HEIGHT = 144
TILE_WIDTH = 144
TILE_DEPTH = 144
NUMBER_INPUT_CHANNELS = 2
NUMBER_OUTPUT_CLASSES = 3

TRAIN_TEST_SPLIT = 0.825  # 0.9
VALIDATE_TEST_SPLIT = 0.50
FILE_RATIO = 1

# 1e-3 for SSol
initial_lr = 1e-3
initial_lr_divisor = 4

RANDOM_SEED = 830  # 826
SAVED_MODEL_NAME = "saved_transfer_model"
EPOCHS = 100
# input
hard_normalizing_input = True
input_with_E_inc = False
input_with_E_angle = False
shuffle_during_training = True

# path
train_input_path = "./data/X_DATA_PATH/"  # "./train_input_path/"
train_output_path = "./data/Y_DATA_PATH/"  # "./train_output_path/"
val_input_path = "./data/X_DATA_PATH/"  # "./val_input_path/"
val_output_path = "./data/Y_DATA_PATH/"  # "./val_output_path/"
test_input_path = "./data/X_DATA_PATH/"  # "./test_input_path/"
test_output_path = "./data/Y_DATA_PATH/"  # "./test_output_path/"

rotation_first = False
rotation = False  # "False" for second coil; "port5" for the right-hand side port of the BC coil
slicing = False  # "True" only if input is [145,145,145] instead of [144,144,144]

# model
out_lay_act = "none"
