import tensorflow as tf  # TensorFlow 2
from tensorflow import keras as K
from dataloader.DataLoader import Dataset

import os
import datetime

from dataloader.argparser import args
from unetmodel.model_mask_att_2_3layer import unet_3d
from unetmodel.loss_functions import rel_err_in_tf, e_field_rms_err

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# tf.config.run_functions_eagerly(True)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

crop_dim = (args.tile_height, args.tile_width,
            args.tile_depth, args.number_input_channels)
model = unet_3d(input_dim=crop_dim, filters=args.filters,
                number_output_classes=args.number_output_classes,
                use_upsampling=args.use_upsampling,
                concat_axis=-1, model_name=args.saved_model_name)

local_opt = K.optimizers.Adam(learning_rate=args.initial_lr)
model.compile(loss=tf.losses.mean_squared_error,
              metrics=[rel_err_in_tf, e_field_rms_err],
              optimizer=local_opt)

checkpoint = K.callbacks.ModelCheckpoint(os.path.join(args.saved_model_name, '{epoch:02d}-{val_loss:.6f}'),
                                         verbose=1,
                                         # save_best_only=True
                                         )

# TensorBoard
logs_dir = os.path.join(args.saved_model_name,
                        "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_logs = K.callbacks.TensorBoard(log_dir=logs_dir)

# Reduced learning rate
reduce_lr = K.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, verbose=1,
                                          patience=3, min_delta=1e-6, min_lr=1e-20)
callbacks = [checkpoint, tb_logs, reduce_lr]

# Randomized dataset into train, validation, test
dataset = Dataset(input_path=args.x_data_path, output_path=args.y_data_path,
                  train_ratio=args.train_test_split, val_ratio=args.validate_test_split,
                  random_seed=args.random_seed,
                  batch_size=args.batch_size, buffer_size=100,
                  input_with_E_inc=args.input_with_E_inc, hard_normalizing_input=args.hard_normalizing_input,
                  file_ratio=args.file_ratio)
dataset.shuffle_train_files()
train_dataset, val_dataset, test_dataset = dataset.get_dataset()
model.fit(x=train_dataset, validation_data=val_dataset,
          epochs=args.epochs,
          callbacks=callbacks,
          verbose=1)
