import tensorflow as tf
import numpy as np
import os
import random


class Dataset:
    def __init__(self, train_input_path, train_output_path,
                 val_input_path, val_output_path,
                 test_input_path, test_output_path,
                 random_seed=645,
                 batch_size=4, buffer_size=100,
                 input_with_E_inc=False, input_with_E_angle=False,
                 hard_normalizing_input=True,
                 shuffle_during_training=True,
                 rotation_first=True, slicing=True, rotation='port5'
                 ):
        self.train_input_path = train_input_path
        self.train_output_path = train_output_path
        self.val_input_path = val_input_path
        self.val_output_path = val_output_path
        self.test_input_path = test_input_path
        self.test_output_path = test_output_path
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.rotation_first = rotation_first
        self.rotation = rotation
        self.slicing = slicing

        self.random_seed = random_seed
        self.train_files = None
        self.val_files = None
        self.test_files = None

        self.input_shape = None
        self.output_shape = None

        self.hard_normalizing_input = hard_normalizing_input
        self.shuffle_during_training = shuffle_during_training
        self.input_with_E_inc = input_with_E_inc
        self.input_with_E_angle = input_with_E_angle
        if sum([self.input_with_E_inc, self.input_with_E_angle]) > 1:
            raise ValueError("At most one variable should be True.")
        if self.input_with_E_inc:
            self.E_field_data = np.load('./data/full_Einc_abs_vox.npy')
        elif self.input_with_E_angle:
            self.E_field_data = np.load('./data/full_Einc_angle_vox.npy')

        train_input_files = os.listdir(self.train_input_path)
        train_output_files = os.listdir(self.train_output_path)
        val_input_files = os.listdir(self.val_input_path)
        val_output_files = os.listdir(self.val_output_path)
        test_input_files = os.listdir(self.test_input_path)
        test_output_files = os.listdir(self.test_output_path)

        train_common_files = set(train_input_files).intersection(set(train_output_files))
        val_common_files = set(val_input_files).intersection(set(val_output_files))
        test_common_files = set(test_input_files).intersection(set(test_output_files))

        self.train_files = list(train_common_files)
        self.val_files = list(val_common_files)
        self.test_files = list(test_common_files)

        # num_files = len(input_files)
        # num_files_per_model = 500
        # num_headmodel = num_files / num_files_per_model
        # if not num_headmodel.is_integer():
        #     raise ValueError('Something wrong with the number of files/head models.')
        # else:
        #     num_headmodel = int(num_headmodel)
        # headmodel_list = list(range(num_headmodel))
        # if self.file_ratio != 1:
        #     shrink_size = int(num_headmodel * self.file_ratio)
        #     headmodel_list = headmodel_list[:shrink_size]
        #
        # random.seed(self.random_seed)
        # random.shuffle(headmodel_list)
        # train_end = int(num_headmodel * self.train_ratio)
        # val_end = int(num_headmodel * (self.train_ratio + self.val_ratio))
        # train_models = headmodel_list[:train_end]
        # val_models = headmodel_list[train_end:val_end]
        # test_models = headmodel_list[val_end:]
        # self.train_files = [f"{str(i).zfill(5)}.npz" for element in train_models
        #                     for i in range(element * 500 + 1, (element + 1) * 500 + 1)]
        # self.val_files = [f"{str(i).zfill(5)}.npz"
        #                   for element in val_models for i in range(element * 500 + 1, (element + 1) * 500 + 1)]
        # self.test_files = [f"{str(i).zfill(5)}.npz"
        #                    for element in test_models for i in range(element * 500 + 1, (element + 1) * 500 + 1)]

        self.train_files_in = [os.path.join(self.train_input_path, name) for name in self.train_files]
        self.train_files_out = [os.path.join(self.train_output_path, name) for name in self.train_files]
        self.val_files_in = [os.path.join(self.val_input_path, name) for name in self.val_files]
        self.val_files_out = [os.path.join(self.val_output_path, name) for name in self.val_files]
        self.test_files_in = [os.path.join(self.test_input_path, name) for name in self.test_files]
        self.test_files_out = [os.path.join(self.test_output_path, name) for name in self.test_files]

    def load_file(self, file_in, file_out):
        input_data_temp = np.load(file_in.numpy().decode("utf-8"))  # convert bytes to string if necessary
        input_data = input_data_temp[input_data_temp.files[0]]
        if self.hard_normalizing_input:
            input_data[..., 0] = input_data[..., 0] / 87.24
            input_data[..., 1] = input_data[..., 1] / 2.664
        if self.input_with_E_inc or self.input_with_E_angle:
            mask = input_data[..., 1:2] > 0
            mask_full = np.repeat(mask, 3, axis=-1)
            input_data = np.concatenate([input_data, np.multiply(self.E_field_data, mask_full)], axis=-1)

        output_data_temp = np.load(file_out.numpy().decode("utf-8"))  # convert bytes to string if necessary
        output_data = output_data_temp[output_data_temp.files[0]]
        if self.rotation or self.slicing:
            input_data_final = self.preprocess_rotate_slice(input_data)
            output_data_final = self.preprocess_rotate_slice(output_data)
        else:
            input_data_final = input_data
            output_data_final = output_data
        return input_data_final, output_data_final

    def preprocess_rotate_slice(self, in_data):
        if self.rotation_first:
            if self.rotation == "port5":
                in_data = np.stack([np.rot90(in_data[:, :, :, i], k=1, axes=(0, 1))
                                    for i in range(in_data.shape[-1])], axis=-1)  # Rotate 90 degrees
            if self.slicing:
                in_data = in_data[:-1, :-1, :-1, :]
        else:
            if self.slicing:
                in_data = in_data[:-1, :-1, :-1, :]
            if self.rotation == "port5":
                in_data = np.stack([np.rot90(in_data[:, :, :, i], k=1, axes=(0, 1))
                                    for i in range(in_data.shape[-1])], axis=-1)  # Rotate 90 degrees
        return in_data

    def read_dataset(self, file_in, file_out):
        input_data, output_data = tf.py_function(func=self.load_file, inp=[file_in, file_out],
                                                 Tout=(tf.float32, tf.float32))
        yield input_data, output_data

    def read_dataset_map(self, file_in, file_out):
        input_data, output_data = tf.py_function(func=self.load_file, inp=[file_in, file_out],
                                                 Tout=(tf.float32, tf.float32))
        return input_data, output_data

    def set_shape(self, file_in, file_out):
        input_data, output_data = tf.py_function(func=self.load_file, inp=[file_in, file_out],
                                                 Tout=(tf.float32, tf.float32))
        self.input_shape = input_data.shape[0:]
        self.output_shape = output_data.shape[0:]

    def get_dataset(self):
        if self.input_shape is None or self.output_shape is None:
            self.set_shape(self.train_files_in[0], self.train_files_out[0])
        train_dataset_in = tf.data.Dataset.list_files(self.train_files_in, shuffle=False)
        train_dataset_out = tf.data.Dataset.list_files(self.train_files_out, shuffle=False)
        train_dataset = tf.data.Dataset.zip((train_dataset_in, train_dataset_out)).shuffle(len(self.train_files))
        train_dataset = train_dataset.map(self.read_dataset_map, num_parallel_calls=tf.data.AUTOTUNE)
        # train_dataset = train_dataset.interleave(
        #     map_func=lambda x, y: tf.data.Dataset.from_generator(self.read_dataset, args=(x, y),
        #                                                          output_signature=(
        #                                                              tf.TensorSpec(shape=self.input_shape,
        #                                                                            dtype=tf.float32),
        #                                                              tf.TensorSpec(shape=self.output_shape,
        #                                                                            dtype=tf.float32))),
        #     cycle_length=tf.data.AUTOTUNE,
        #     num_parallel_calls=tf.data.AUTOTUNE)

        if self.shuffle_during_training:
            train_dataset = train_dataset.shuffle(buffer_size=self.buffer_size)
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        val_dataset_in = tf.data.Dataset.from_tensor_slices(self.val_files_in)
        val_dataset_out = tf.data.Dataset.from_tensor_slices(self.val_files_out)
        val_dataset = tf.data.Dataset.zip((val_dataset_in, val_dataset_out))
        val_dataset = val_dataset.map(self.read_dataset_map, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(self.batch_size)
        val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        test_dataset_in = tf.data.Dataset.from_tensor_slices(self.test_files_in)
        test_dataset_out = tf.data.Dataset.from_tensor_slices(self.test_files_out)
        test_dataset = tf.data.Dataset.zip((test_dataset_in, test_dataset_out))
        test_dataset = test_dataset.map(self.read_dataset_map, num_parallel_calls=tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(self.batch_size)
        test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_dataset, val_dataset, test_dataset

    def shuffle_train_files(self):
        random.seed(self.random_seed)
        random.shuffle(self.train_files)
        self.train_files_in = [os.path.join(self.train_input_path, name) for name in self.train_files]
        self.train_files_out = [os.path.join(self.train_output_path, name) for name in self.train_files]

    def train_list(self):
        return self.train_files

    def val_list(self):
        return self.val_files

    def test_list(self):
        return self.test_files
