from dataloader.argparser import args
import tensorflow as tf
from tensorflow import keras as K


def unet_3d(input_dim, filters=args.filters,
            number_output_classes=args.number_output_classes,
            use_upsampling=args.use_upsampling,
            concat_axis=-1, model_name=args.saved_model_name,
            out_lay_act=args.out_lay_act):
    """
    3D U-Net
    """

    def ConvolutionBlock(x, name, filters, params):
        """
        Convolutional block of layers
        Per the original paper this is back to back 3D convs
        with batch norm and then ReLU.
        """

        x = K.layers.Conv3D(filters=filters, **params, name=name + "_conv0")(x)
        x = K.layers.BatchNormalization(name=name + "_bn0")(x)
        x = K.layers.Activation("relu", name=name + "_relu0")(x)

        x = K.layers.Conv3D(filters=filters, **params, name=name + "_conv1")(x)
        x = K.layers.BatchNormalization(name=name + "_bn1")(x)
        x = K.layers.Activation("relu", name=name + "_relu1")(x)

        x = K.layers.Conv3D(filters=filters, **params, name=name + "_conv2")(x)
        x = K.layers.BatchNormalization(name=name + "_bn2")(x)
        x = K.layers.Activation("relu", name=name + "_relu2")(x)

        return x

    inputs = K.layers.Input(shape=input_dim, name="MRImages")

    params = dict(kernel_size=(3, 3, 3), activation=None,
                  padding="same",
                  kernel_initializer="he_uniform")

    # Transposed convolution parameters
    params_trans = dict(kernel_size=(2, 2, 2), strides=(2, 2, 2),
                        padding="same",
                        kernel_initializer="he_uniform")

    # BEGIN - Encoding path
    encodeA = ConvolutionBlock(inputs, "encodeA", filters, params)
    poolA = K.layers.MaxPooling3D(name="poolA", pool_size=(2, 2, 2))(encodeA)

    encodeB = ConvolutionBlock(poolA, "encodeB", filters * 2, params)
    poolB = K.layers.MaxPooling3D(name="poolB", pool_size=(2, 2, 2))(encodeB)

    encodeC = ConvolutionBlock(poolB, "encodeC", filters * 4, params)
    poolC = K.layers.MaxPooling3D(name="poolC", pool_size=(2, 2, 2))(encodeC)

    encodeD = ConvolutionBlock(poolC, "encodeD", filters * 8, params)
    poolD = K.layers.MaxPooling3D(name="poolD", pool_size=(2, 2, 2))(encodeD)

    encodeE = ConvolutionBlock(poolD, "encodeE", filters * 16, params)
    # END - Encoding path

    # BEGIN - Decoding path
    if use_upsampling:
        up = K.layers.UpSampling3D(name="upE", size=(2, 2, 2))(encodeE)
    else:
        up = K.layers.Conv3DTranspose(name="transconvE", filters=filters * 8,
                                      **params_trans)(encodeE)
    concatD = K.layers.concatenate(
        [up, encodeD], axis=concat_axis, name="concatD")

    decodeC = ConvolutionBlock(concatD, "decodeC", filters * 8, params)

    if use_upsampling:
        up = K.layers.UpSampling3D(name="upC", size=(2, 2, 2))(decodeC)
    else:
        up = K.layers.Conv3DTranspose(name="transconvC", filters=filters * 4,
                                      **params_trans)(decodeC)
    concatC = K.layers.concatenate(
        [up, encodeC], axis=concat_axis, name="concatC")

    decodeB = ConvolutionBlock(concatC, "decodeB", filters * 4, params)

    if use_upsampling:
        up = K.layers.UpSampling3D(name="upB", size=(2, 2, 2))(decodeB)
    else:
        up = K.layers.Conv3DTranspose(name="transconvB", filters=filters * 2,
                                      **params_trans)(decodeB)
    concatB = K.layers.concatenate(
        [up, encodeB], axis=concat_axis, name="concatB")

    decodeA = ConvolutionBlock(concatB, "decodeA", filters * 2, params)

    if use_upsampling:
        up = K.layers.UpSampling3D(name="upA", size=(2, 2, 2))(decodeA)
    else:
        up = K.layers.Conv3DTranspose(name="transconvA", filters=filters,
                                      **params_trans)(decodeA)
    concatA = K.layers.concatenate(
        [up, encodeA], axis=concat_axis, name="concatA")

    # END - Decoding path
    # Multiscale

    convOut1 = K.layers.Conv3D(filters=filters, **params, name='convOut1' + "_conv0")(concatA)
    convOut1 = K.layers.BatchNormalization(name='convOut1' + "_bn0")(convOut1)
    convOut1 = K.layers.Activation("relu", name='convOut1' + "_relu0")(convOut1)
    convOut1 = K.layers.Conv3D(filters=number_output_classes, **params, name='convOut1' + "_conv1")(convOut1)
    convOut1 = K.layers.BatchNormalization(name='convOut1' + "_bn1")(convOut1)
    convOut1 = K.layers.Activation("relu", name='convOut1' + "_relu1")(convOut1)

    decodeA_1X1X1 = K.layers.Conv3D(name="decodeA_1X1X1_conv",
                                    filters=number_output_classes, kernel_size=(1, 1, 1),
                                    activation=None)(decodeA)
    decodeA_1X1X1 = K.layers.BatchNormalization(name="decodeA_1X1X1" + '_BN')(decodeA_1X1X1)

    decodeB_up = K.layers.Conv3D(name="decodeB_up_conv",
                                 filters=number_output_classes, kernel_size=(1, 1, 1),
                                 activation=None)(decodeB)
    decodeB_up = K.layers.BatchNormalization(name="decodeB_up" + '_BN0')(decodeB_up)
    decodeB_up = K.layers.Conv3DTranspose(name="decodeB_up_deconv", filters=number_output_classes,
                                          **params_trans)(decodeB_up)
    decodeB_up = K.layers.BatchNormalization(name="decodeB_up" + '_BN1')(decodeB_up)

    addOut2 = decodeA_1X1X1 + decodeB_up
    convOut2_up = K.layers.Conv3DTranspose(name="convOut2_up", filters=number_output_classes,
                                           **params_trans)(addOut2)
    addOut = convOut1 + convOut2_up

    if out_lay_act == "none":
        prediction = K.layers.Conv3D(name="PredictionMask",
                                     filters=number_output_classes,
                                     kernel_size=(1, 1, 1),
                                     activation=None)(addOut)
    elif out_lay_act == "sig":
        prediction = K.layers.Conv3D(name="PredictionMask",
                                     filters=number_output_classes,
                                     kernel_size=(1, 1, 1),
                                     activation="sigmoid")(addOut)
    elif out_lay_act == 'tanh':
        prediction = K.layers.Conv3D(name="PredictionMask",
                                     filters=number_output_classes,
                                     kernel_size=(1, 1, 1),
                                     activation="tanh")(addOut)
    elif out_lay_act == 'softmax':
        prediction = K.layers.Conv3D(name="PredictionMask",
                                     filters=number_output_classes,
                                     kernel_size=(1, 1, 1),
                                     activation="softmax")(addOut)
    else:
        prediction = K.layers.Conv3D(name="PredictionMask",
                                     filters=number_output_classes,
                                     kernel_size=(1, 1, 1),
                                     activation=None)(addOut)

    if True:
        mask_temp = tf.math.greater(inputs, 0)
        mask = mask_temp[..., 1:2]  # Conductivity map
        # print(mask.shape)
        mask = tf.repeat(mask, tf.shape(prediction)[-1], axis=-1)
        mask = tf.cast(mask, tf.float32)
        prediction = K.layers.Multiply()([mask, prediction])

    model = K.models.Model(inputs=[inputs], outputs=[prediction],
                           name=model_name)

    if args.print_model:
        model.summary()

    return model
