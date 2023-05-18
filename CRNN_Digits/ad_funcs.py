import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    LSTM,
    Dense,
    TimeDistributed,
    Reshape,
)

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical


# Function to load and preprocess images
def preprocess_image(image_path, target_size=(32, 32)):
    img = load_img(image_path, color_mode="grayscale", target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return img_array


# def build_crnn_model(input_shape=(32, 32, 1), num_classes=11):
#     """
#     Build a CRNN model with the given input shape and number of output classes.

#     Args:
#         input_shape (tuple): The input shape of the images (height, width, channels).
#         num_classes (int): The number of output classes (digits).

#     Returns:
#         model (Model): The compiled CRNN model.
#     """

#     # Input layer
#     inputs = Input(shape=input_shape)

#     # Convolutional Layers
#     conv1 = Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same")(inputs)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

#     conv2 = Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")(pool1)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

#     # Prepare for LSTM
#     reshaped = Reshape(
#         target_shape=((input_shape[0] // 4) * (input_shape[1] // 4), 64)
#     )(pool2)

#     # Recurrent Layers
#     lstm1 = LSTM(128, return_sequences=True)(reshaped)
#     lstm2 = LSTM(128, return_sequences=True)(lstm1)

#     # Fully Connected Layer
#     dense = TimeDistributed(Dense(num_classes, activation="softmax"))(lstm2)

#     # Define the model
#     model = Model(inputs=inputs, outputs=dense)

#     # Compile the model
#     model.compile(
#         optimizer="adam",
#         loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
#     )

#     return model


def build_crnn_model(input_shape=(32, 32, 1), num_classes=11):
    """
    Build a CRNN model with the given input shape and number of output classes.

    Args:
        input_shape (tuple): The input shape of the images (height, width, channels).
        num_classes (int): The number of output classes (digits).

    Returns:
        model (Model): The compiled CRNN model.
    """

    # Input layer
    inputs = Input(shape=input_shape)

    # Convolutional Layers
    conv1 = Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same")(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Prepare for LSTM
    reshaped = Reshape(
        target_shape=((input_shape[0] // 4) * (input_shape[1] // 4), 64)
    )(pool2)

    # Recurrent Layers
    lstm1 = LSTM(128, return_sequences=True)(reshaped)
    lstm2 = LSTM(128, return_sequences=False)(lstm1)

    # Fully Connected Layer
    dense = Dense(num_classes, activation="softmax")(lstm2)

    # Define the model
    model = Model(inputs=inputs, outputs=dense)

    # Compile the model
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    )

    return model
