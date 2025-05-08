from dataloader import settings_transfer
import argparse

settings = settings_transfer

parser = argparse.ArgumentParser(
    description="Train 3D U-Net model", add_help=True,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--train_input_path",
                    default=settings.train_input_path,
                    help="Root directory for Medical Decathlon dataset")
parser.add_argument("--train_output_path",
                    default=settings.train_output_path,
                    help="Root directory for Medical Decathlon dataset")
parser.add_argument("--val_input_path",
                    default=settings.val_input_path,
                    help="Root directory for Medical Decathlon dataset")
parser.add_argument("--val_output_path",
                    default=settings.val_output_path,
                    help="Root directory for Medical Decathlon dataset")
parser.add_argument("--test_input_path",
                    default=settings.test_input_path,
                    help="Root directory for Medical Decathlon dataset")
parser.add_argument("--test_output_path",
                    default=settings.test_output_path,
                    help="Root directory for Medical Decathlon dataset")
parser.add_argument("--rotation_first",
                    default=settings.rotation_first,
                    help="Rotation first or Slicing?")
parser.add_argument("--rotation",
                    default=settings.rotation,
                    help="Activation port.")
parser.add_argument("--slicing",
                    default=settings.slicing,
                    help="Get rid of the unnecessary voxel")


parser.add_argument("--epochs",
                    type=int,
                    default=settings.EPOCHS,
                    help="Number of epochs")
parser.add_argument("--saved_model_name",
                    default=settings.SAVED_MODEL_NAME,
                    help="Save model to this path")
parser.add_argument("--batch_size",
                    type=int,
                    default=settings.BATCH_SIZE,
                    help="Training batch size")
parser.add_argument("--tile_height",
                    type=int,
                    default=settings.TILE_HEIGHT,
                    help="Size of the 3D patch height")
parser.add_argument("--tile_width",
                    type=int,
                    default=settings.TILE_WIDTH,
                    help="Size of the 3D patch width")
parser.add_argument("--tile_depth",
                    type=int,
                    default=settings.TILE_DEPTH,
                    help="Size of the 3D patch depth")
parser.add_argument("--number_input_channels",
                    type=int,
                    default=settings.NUMBER_INPUT_CHANNELS,
                    help="Number of input channels")
parser.add_argument("--number_output_classes",
                    type=int,
                    default=settings.NUMBER_OUTPUT_CLASSES,
                    help="Number of output classes/channels")
parser.add_argument("--train_test_split",
                    type=float,
                    default=settings.TRAIN_TEST_SPLIT,
                    help="Train/test split (0-1)")
parser.add_argument("--validate_test_split",
                    type=float,
                    default=settings.VALIDATE_TEST_SPLIT,
                    help="Validation/test split (0-1)")
parser.add_argument("--print_model",
                    action="store_true",
                    default=settings.PRINT_MODEL,
                    help="Print the summary of the model layers")
parser.add_argument("--filters",
                    type=int,
                    default=settings.FILTERS,
                    help="Number of filters in the first convolutional layer")
parser.add_argument("--use_upsampling",
                    action="store_true",
                    default=settings.USE_UPSAMPLING,
                    help="Use upsampling instead of transposed convolution")
parser.add_argument("--random_seed",
                    default=settings.RANDOM_SEED,
                    help="Random seed for determinism")
parser.add_argument("--initial_lr",
                    default=settings.initial_lr,
                    help="Initial learning rate for model training")
parser.add_argument("--initial_lr_divisor",
                    default=settings.initial_lr_divisor,
                    help="Initial learning rate divisor for transfer learning")

parser.add_argument("--hard_normalizing_input",
                    default=settings.hard_normalizing_input,
                    help="Hard normalizing input maps, cond and perm")
parser.add_argument("--input_with_E_inc",
                    default=settings.input_with_E_inc,
                    help="add E incident as inputs as well")
parser.add_argument("--input_with_E_angle",
                    default=settings.input_with_E_angle,
                    help="add E angle as inputs as well")


parser.add_argument("--out_lay_act",
                    default=settings.out_lay_act,
                    help="Output layer activation function type, none, sig, tanh.")

parser.add_argument("--file_ratio",
                    default=settings.FILE_RATIO,
                    help="Control the size of datasets.")

parser.add_argument("--shuffle_during_training",
                    default=settings.shuffle_during_training,
                    help="Shuffle files between each epoch.")


args, unknown = parser.parse_known_args()
# args = parser.parse_args()
