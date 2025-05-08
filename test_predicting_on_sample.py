import tensorflow as tf  # TensorFlow 2
from tensorflow import keras as K
from dataloader.DataLoader_folder import Dataset

import os

from dataloader.argparser import args
from unetmodel.loss_functions import rel_err_in_tf, e_field_rms_err

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# tf.config.run_functions_eagerly(True)

model_folder = './data/model/45-0.227051/'
model = K.models.load_model(model_folder, compile=False)
current_lr = K.backend.eval(model.optimizer.lr)
local_opt = K.optimizers.Adam(learning_rate=current_lr)
model.compile(loss=tf.losses.mean_squared_error,
              metrics=[rel_err_in_tf, e_field_rms_err,
                       ],
              optimizer=local_opt)

dataset = Dataset(train_input_path=args.x_data_path,
                  train_output_path=args.y_data_path,
                  val_input_path=args.x_data_path,
                  val_output_path=args.y_data_path,
                  test_input_path=args.x_data_path,
                  test_output_path=args.y_data_path,
                  random_seed=args.random_seed,
                  batch_size=1, buffer_size=100,
                  input_with_E_inc=args.input_with_E_inc, hard_normalizing_input=args.hard_normalizing_input,
                  rotation_first=False, slicing=False, rotation=False)

train_dataset, val_dataset, test_dataset = dataset.get_dataset()
X, y = next(iter(val_dataset.take(1)))
X, y = X.numpy(), y.numpy()
y_pre = model.predict(X)
print(np.mean(np.abs((np.sum(y ** 2, axis=-1) - np.sum(y_pre ** 2, axis=-1)) / (np.sum(y ** 2, axis=-1) + 1e-16))[
                  np.sum(y, axis=-1) != 0]))
