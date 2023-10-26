import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, \
    UpSampling2D, concatenate, Input, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.models import Model
import random

# Seed the random number generators for reproducibility
random_seed = 42
random.seed(random_seed)
tf.random.set_seed(random_seed)
np.random.seed(random_seed)

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, sublayers=2):
    """
    Crée un bloc de couches Conv2D avec option de normalisation BatchNormalization.
    """
    for idx in range(sublayers):
        conv = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                   kernel_initializer="he_normal", padding="same")(input_tensor if idx == 0 else conv)
        if batchnorm:
            conv = BatchNormalization()(conv)
        conv = Activation("relu")(conv)

    return conv


def conv2d_transpose_block(input_tensor, concatenate_tensor, n_filters, kernel_size=3, strides=2, transpose=False):
    """
    Crée un bloc de couches Conv2DTranspose (ou UpSampling2D) avec option de concaténation.
    """
    if transpose:
        conv = Conv2DTranspose(n_filters, (kernel_size, kernel_size),
                               strides=(strides, strides), padding='same')(input_tensor)
    else:
        conv = Conv2D(n_filters, (kernel_size, kernel_size), activation='relu', padding='same',
                      kernel_initializer='he_normal')(UpSampling2D(size=(kernel_size, kernel_size))(input_tensor))
    conv = Activation("relu")(conv)
    concatenation = concatenate([conv, concatenate_tensor])

    return concatenation


def build_unet(input_shape=(512, 512, 3), filters=[16, 32, 64, 128, 256], batchnorm=True, transpose=False, dropout_flag=False):
    """
    Construit un modèle U-Net en fonction des paramètres spécifiés.
    """
    conv_dict = dict()
    inputs = Input(input_shape)
    dropout_rate = 0.5

    for idx, n_filters in enumerate(filters[:-1]):
        conv = conv2d_block(inputs if n_filters == filters[0] else max_pool,
                            n_filters=n_filters, kernel_size=3,
                            batchnorm=batchnorm)
        max_pool = MaxPooling2D((2, 2))(conv)
        conv_dict[f"conv2d_{idx+1}"] = conv

    conv_middle = conv2d_block(max_pool, n_filters=filters[-1], kernel_size=3, batchnorm=batchnorm)

    for idx, n_filters in enumerate(reversed(filters[:-1])):
        concatenation = conv2d_transpose_block(conv_middle if idx == 0 else conv,
                                               conv_dict[f"conv2d_{len(conv_dict) - idx}"],
                                               n_filters, kernel_size=2, strides=2, transpose=transpose)

        conv = conv2d_block(concatenation, n_filters=n_filters, kernel_size=3,
                            batchnorm=batchnorm)
    outputs = Conv2D(3, (1, 1), activation='softmax')(conv)
    model = Model(inputs=inputs, outputs=outputs)

    return model
