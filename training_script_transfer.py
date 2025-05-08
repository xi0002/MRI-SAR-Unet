import tensorflow as tf  # TensorFlow 2
from tensorflow import keras as K
from dataloader.DataLoader_folder import Dataset

import os
import datetime

from dataloader.argparser_transfer import args
from unetmodel.loss_functions import rel_err_in_tf, e_field_rms_err

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# tf.config.run_functions_eagerly(True)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model_folder = './data/transfer_base_model_no_Einc/42-0.250747/'
model = K.models.load_model(model_folder, compile=False)
current_lr = K.backend.eval(model.optimizer.lr)
local_opt = K.optimizers.Adam(learning_rate=args.initial_lr/args.initial_lr_divisor)
model.compile(loss=tf.losses.mean_squared_error,
              metrics=[rel_err_in_tf, e_field_rms_err,
                       ],
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
dataset = Dataset(train_input_path=args.train_input_path,
                  train_output_path=args.train_output_path,
                  val_input_path=args.val_input_path,
                  val_output_path=args.val_output_path,
                  test_input_path=args.test_input_path,
                  test_output_path=args.test_output_path,
                  random_seed=args.random_seed,
                  batch_size=args.batch_size, buffer_size=100,
                  input_with_E_inc=args.input_with_E_inc, hard_normalizing_input=args.hard_normalizing_input,
                  rotation_first=args.rotation_first, slicing=args.slicing, rotation=args.rotation)
dataset.shuffle_train_files()
train_dataset, val_dataset, test_dataset = dataset.get_dataset()
model.fit(x=train_dataset, validation_data=val_dataset,
          epochs=args.epochs,
          callbacks=callbacks,
          verbose=1)
