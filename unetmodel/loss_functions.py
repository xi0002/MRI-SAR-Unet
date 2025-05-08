import tensorflow as tf
import numpy as np


def rel_err(target, prediction):  # metric
    tg = target.numpy()
    pd = prediction.numpy()
    idx_nonzero = np.nonzero(tg)
    tg_ = tg[idx_nonzero]
    pd_ = pd[idx_nonzero]
    relative_err = np.abs(tg_ - pd_) / np.abs(tg_)
    return np.mean(relative_err)


def rel_err_for_testing(target, prediction):
    # Same as above, avoid converting ndarray to numpy
    # might use it when generating errors for each testing point
    tg = target
    pd = prediction
    idx_nonzero = np.nonzero(tg)
    tg_ = tg[idx_nonzero]
    pd_ = pd[idx_nonzero]
    relative_err = np.abs(tg_ - pd_) / np.abs(tg_)
    return np.mean(relative_err)


def mse_add_mape(weight_mse, weight_mape):
    def mse_add_mape_inner(target, prediction):
        err = weight_mse * tf.losses.mean_squared_error(target, prediction) +\
              weight_mape * tf.losses.mean_absolute_percentage_error(target, prediction)
        return err
    return mse_add_mape_inner


def mse_mape(target, prediction):
    err = tf.losses.mean_squared_error(target, prediction) * \
          tf.losses.mean_absolute_percentage_error(target, prediction)
    return err


def mae_mape(target, prediction):
    err = tf.losses.mean_absolute_error(target, prediction) * \
          tf.losses.mean_absolute_percentage_error(target, prediction)
    return err


def rel_err_in_tf(target, prediction):
    # only on tissue voxels
    nonzero_mask = tf.where(target != 0)
    tg = tf.gather_nd(target, nonzero_mask)
    pd = tf.gather_nd(prediction, nonzero_mask)
    relative_err = tf.math.abs(tf.math.divide_no_nan(tf.math.subtract(pd, tg), tg))
    return tf.math.reduce_mean(relative_err, axis=-1)


def rel_err_in_tf_voxel_max(target, prediction):
    # only on tissue voxels
    # for Esol
    # instead of using y in denominator
    # using the maximum of 6 values from real,imag parts of Ex,Ey,Ez
    # so each voxel has its only denominator
    nonzero_mask = tf.where(target != 0)
    tg = tf.gather_nd(target, nonzero_mask)  # to vec
    pd = tf.gather_nd(prediction, nonzero_mask)  # to vec
    max_voxel = tf.math.reduce_max(tf.math.abs(target), axis=-1)
    max_voxel_full = tf.repeat(max_voxel[..., None], target.shape[-1], axis=-1)
    max_voxel_full_vec = tf.gather_nd(max_voxel_full, nonzero_mask)  # to vec
    relative_err = tf.math.abs(tf.math.divide_no_nan(tf.math.subtract(pd, tg), max_voxel_full_vec))
    return tf.math.reduce_mean(relative_err, axis=-1)


def e_field_rms_err(target, prediction):
    # only on tissue voxels
    # for Esol
    # calculate E rms for each voxel
    # and find rel err
    target_sq = tf.math.square(target)
    prediction_sq = tf.math.square(prediction)
    target_e_rms = tf.math.sqrt(tf.math.reduce_sum(target_sq, axis=-1)/2)
    prediction_e_rms = tf.math.sqrt(tf.math.reduce_sum(prediction_sq, axis=-1)/2)
    nonzero_mask = tf.where(target_e_rms != 0)
    tg = tf.gather_nd(target_e_rms, nonzero_mask)  # to vec
    pd = tf.gather_nd(prediction_e_rms, nonzero_mask)  # to vec
    relative_err = tf.math.abs(tf.math.divide_no_nan(tf.math.subtract(pd, tg), tg))
    return tf.math.reduce_mean(relative_err, axis=-1)
