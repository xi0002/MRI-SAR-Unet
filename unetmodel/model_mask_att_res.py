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

    def ResidualBlock(x, name, res_filters, params):
        """
        Convolutional block of layers
        Per the original paper this is back to back 3D convs
        with batch norm and then ReLU.
        """
        res_params = params
        # res_params['kernel_size'] = (1, 1, 1)

        xf_qq = 1
        ResNetV1 = 0
        ResNetV2 = 0
        if xf_qq:  # as in Qiqi's paper
            x1_1 = K.layers.Conv3D(filters=res_filters, **params, name=name + "_conv0")(x)
            x1_2 = K.layers.BatchNormalization(name=name + "_bn0")(x1_1)
            x1_3 = K.layers.Activation("relu", name=name + "_relu0")(x1_2)

            x2_1 = K.layers.Conv3D(filters=res_filters, **params, name=name + "_conv1")(x1_3)
            x2_2 = K.layers.BatchNormalization(name=name + "_bn1")(x2_1)
            x2_3 = K.layers.Activation("relu", name=name + "_relu1")(x2_2)

            x3_1 = K.layers.Conv3D(filters=res_filters, **params, name=name + "_conv2")(x2_3)
            x3_2 = K.layers.BatchNormalization(name=name + "_bn2")(x3_1)
            x3_3 = K.layers.Add()([x1_2, x3_2])
            x3_4 = K.layers.Activation("relu", name=name)(x3_3)
            return x3_4
        elif ResNetV1:
            x1 = K.layers.Conv3D(filters=res_filters, **params, name=name + "_conv0")(x)
            x1 = K.layers.BatchNormalization(name=name + "_bn0")(x1)
            x1 = K.layers.Activation("relu", name=name + "_relu0")(x1)

            x1 = K.layers.Conv3D(filters=res_filters, **params, name=name + "_conv1")(x1)
            x1 = K.layers.BatchNormalization(name=name + "_bn1")(x1)

            x_res = K.layers.Conv3D(filters=res_filters, **res_params, name=name + "_conv_res")(x)

            x = K.layers.Add()([x1, x_res])
            x = K.layers.Activation("relu", name=name + "_relu1")(x)
            return x
        elif ResNetV2:
            x1 = K.layers.BatchNormalization(name=name + "_bn0")(x)
            x1 = K.layers.Activation("relu", name=name + "_relu0")(x1)
            x1 = K.layers.Conv3D(filters=res_filters, **params, name=name + "_conv0")(x1)

            x1 = K.layers.BatchNormalization(name=name + "_bn1")(x1)
            x1 = K.layers.Activation("relu", name=name + "_relu1")(x1)
            x1 = K.layers.Conv3D(filters=res_filters, **params, name=name + "_conv1")(x1)

            x_res = K.layers.Conv3D(filters=res_filters, **res_params, name=name + "_conv_res")(x)
            x = K.layers.Add()([x_res, x1])
            return x

    def expend_as(tensor, rep):

        # Anonymous lambda function to expand the specified axis by a factor of argument, rep.
        # If tensor has shape (512,512,N), lambda will return a tensor of shape (512,512,N*rep), if specified axis=2

        my_repeat = K.layers.Lambda(lambda x, repnum: K.backend.repeat_elements(x, repnum, axis=4),
                                    arguments={'repnum': rep})(tensor)
        return my_repeat

    def AttentionBlock(x_upper, x_lower, att_filters):
        x_upper_1 = K.layers.Conv3D(filters=att_filters, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                                    use_bias=False)(x_upper)
        x_upper_2 = K.layers.BatchNormalization()(x_upper_1)

        x_lower_1 = K.layers.Conv3D(filters=att_filters, kernel_size=(1, 1, 1), strides=(1, 1, 1),
                                    use_bias=True)(x_lower)
        x_lower_2 = K.layers.BatchNormalization()(x_lower_1)

        x3 = K.layers.Add()([x_upper_2, x_lower_2])
        x3 = K.layers.Activation('relu')(x3)

        x4 = K.layers.Conv3D(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1))(x3)
        x4 = K.layers.BatchNormalization()(x4)
        x4 = K.layers.Activation("sigmoid")(x4)

        x5 = K.layers.UpSampling3D(size=(2, 2, 2))(x4)
        x5 = K.layers.multiply([x5, x_upper])

        return x5

    inputs = K.layers.Input(shape=input_dim, name="MRImages")

    params = dict(kernel_size=(3, 3, 3), activation=None,
                  padding="same",
                  kernel_initializer="he_uniform")

    # Transposed convolution parameters
    params_trans = dict(kernel_size=(2, 2, 2), strides=(2, 2, 2),
                        padding="same",
                        kernel_initializer="he_uniform")

    # BEGIN - Encoding path
    encodeA = ResidualBlock(inputs, "encodeA", filters, params)
    poolA = K.layers.MaxPooling3D(name="poolA", pool_size=(2, 2, 2))(encodeA)

    encodeB = ResidualBlock(poolA, "encodeB", filters * 2, params)
    poolB = K.layers.MaxPooling3D(name="poolB", pool_size=(2, 2, 2))(encodeB)

    encodeC = ResidualBlock(poolB, "encodeC", filters * 4, params)
    poolC = K.layers.MaxPooling3D(name="poolC", pool_size=(2, 2, 2))(encodeC)

    encodeD = ResidualBlock(poolC, "encodeD", filters * 8, params)
    poolD = K.layers.MaxPooling3D(name="poolD", pool_size=(2, 2, 2))(encodeD)

    encodeE = ResidualBlock(poolD, "encodeE", filters * 16, params)
    # END - Encoding path

    # BEGIN - Decoding path
    if use_upsampling:
        up = K.layers.UpSampling3D(name="upE", size=(2, 2, 2))(encodeE)
    else:
        up = K.layers.Conv3DTranspose(name="transconvE", filters=filters * 8,
                                      **params_trans)(encodeE)
    attnD = AttentionBlock(x_lower=encodeE, x_upper=encodeD, att_filters=filters * 8)
    concatD = K.layers.concatenate(
        [up, attnD], axis=concat_axis, name="concatD")

    decodeC = ResidualBlock(concatD, "decodeC", filters * 8, params)

    if use_upsampling:
        up = K.layers.UpSampling3D(name="upC", size=(2, 2, 2))(decodeC)
    else:
        up = K.layers.Conv3DTranspose(name="transconvC", filters=filters * 4,
                                      **params_trans)(decodeC)
    attnC = AttentionBlock(x_lower=encodeD, x_upper=encodeC, att_filters=filters * 4)
    concatC = K.layers.concatenate(
        [up, attnC], axis=concat_axis, name="concatC")

    decodeB = ResidualBlock(concatC, "decodeB", filters * 4, params)

    if use_upsampling:
        up = K.layers.UpSampling3D(name="upB", size=(2, 2, 2))(decodeB)
    else:
        up = K.layers.Conv3DTranspose(name="transconvB", filters=filters * 2,
                                      **params_trans)(decodeB)
    attnB = AttentionBlock(x_lower=encodeC, x_upper=encodeB, att_filters=filters * 2)
    concatB = K.layers.concatenate(
        [up, attnB], axis=concat_axis, name="concatB")

    decodeA = ResidualBlock(concatB, "decodeA", filters * 2, params)

    if use_upsampling:
        up = K.layers.UpSampling3D(name="upA", size=(2, 2, 2))(decodeA)
    else:
        up = K.layers.Conv3DTranspose(name="transconvA", filters=filters,
                                      **params_trans)(decodeA)
    attnA = AttentionBlock(x_lower=encodeB, x_upper=encodeA, att_filters=filters)
    concatA = K.layers.concatenate(
        [up, attnA], axis=concat_axis, name="concatA")

    # END - Decoding path

    convOut = ResidualBlock(concatA, "convOut", filters, params)

    if out_lay_act == "none":
        prediction = K.layers.Conv3D(name="PredictionMask",
                                     filters=number_output_classes,
                                     kernel_size=(1, 1, 1),
                                     activation=None)(convOut)
    elif out_lay_act == "sig":
        prediction = K.layers.Conv3D(name="PredictionMask",
                                     filters=number_output_classes,
                                     kernel_size=(1, 1, 1),
                                     activation="sigmoid")(convOut)
    elif out_lay_act == 'tanh':
        prediction = K.layers.Conv3D(name="PredictionMask",
                                     filters=number_output_classes,
                                     kernel_size=(1, 1, 1),
                                     activation="tanh")(convOut)
    elif out_lay_act == 'softmax':
        prediction = K.layers.Conv3D(name="PredictionMask",
                                     filters=number_output_classes,
                                     kernel_size=(1, 1, 1),
                                     activation="softmax")(convOut)
    else:
        prediction = K.layers.Conv3D(name="PredictionMask",
                                     filters=number_output_classes,
                                     kernel_size=(1, 1, 1),
                                     activation=None)(convOut)

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
