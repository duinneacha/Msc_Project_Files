import pandas as pd
import numpy as np
import cv2
import os
import scipy.io
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    Reshape,
    Softmax,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


import sys

sys.path.append("../project_functions/")
import ad_functions as adfns


############################################################################################################
def load_svhn_data(train_mat_file, test_mat_file):
    train_data = scipy.io.loadmat(train_mat_file)
    test_data = scipy.io.loadmat(test_mat_file)

    X_train = train_data["X"]
    y_train = train_data["y"]
    X_test = test_data["X"]
    y_test = test_data["y"]

    X_train = np.transpose(X_train, (3, 0, 1, 2))
    X_test = np.transpose(X_test, (3, 0, 1, 2))

    # Replace the label 10 with 0, as the dataset uses 10 for the digit 0
    y_train = np.where(y_train == 10, 0, y_train)
    y_test = np.where(y_test == 10, 0, y_test)

    return X_train, y_train, X_test, y_test


############################################################################################################
def preprocess_labels(labels, max_digits=5, blank_class=10):
    processed_labels = []
    for label in labels:
        digits = [int(d) for d in str(label[0])]
        while len(digits) < max_digits:
            digits.append(blank_class)
        processed_labels.append(digits)

    return to_categorical(np.array(processed_labels), num_classes=11)


############################################################################################################
def create_crnn_model(
    input_shape=(32, 32, 3), num_digits=5, num_classes=11, learning_rate=1e-4
):
    input_layer = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(num_classes * num_digits, activation="linear")(x)
    x = Reshape((num_digits, num_classes))(x)
    x = Softmax(axis=-1)(x)

    model = Model(inputs=input_layer, outputs=x)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


############################################################################################################
def plot_loss(history_dict):

    # Loss plot
    plt.figure(figsize=(12, 6))
    plt.plot(history_dict["loss"], label="Training Loss")
    plt.plot(history_dict["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()


############################################################################################################
def plot_accuracy(history_dict):

    # Accuracy plot
    plt.figure(figsize=(12, 6))
    plt.plot(history_dict["accuracy"], label="Training Accuracy")
    plt.plot(history_dict["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.show()


############################################################################################################
def preprocess_image(image_path_or_image):
    if isinstance(image_path_or_image, str):
        image = cv2.imread(image_path_or_image)
    else:
        image = image_path_or_image

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    return image


############################################################################################################
def extract_digits(processed_image):
    # Find contours in the binary image
    contours, _ = cv2.findContours(
        processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Initialize a list to store the bounding rectangles of each digit
    digit_rects = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 100:  # Filter out small/noisy contours
            digit_rects.append((x, y, w, h))

    # Sort the digit rectangles by their x-coordinate (left-to-right)
    digit_rects = sorted(digit_rects, key=lambda x: x[0])

    # Extract individual digits using the bounding rectangles
    digit_images = [processed_image[y : y + h, x : x + w] for x, y, w, h in digit_rects]

    return digit_images


############################################################################################################
# def preprocess_digit(digit_image, target_size=(32, 32)):
#     digit_image = cv2.resize(digit_image, target_size)
#     digit_image = digit_image.astype("float32") / 255.0
#     digit_image = np.stack(
#         (digit_image,) * 3, axis=-1
#     )  # Replicate the channel three times
#     digit_image = np.expand_dims(digit_image, axis=0)
#     return digit_image


def preprocess_digit(digit_image, target_size=(32, 32)):
    digit_image = cv2.resize(digit_image, target_size)
    digit_image = digit_image.astype("float32") / 255.0
    # Remove the following line:
    # digit_image = np.stack((digit_image,) * 3, axis=-1)
    digit_image = np.expand_dims(digit_image, axis=-1)  # Add the channel axis
    digit_image = np.expand_dims(digit_image, axis=0)  # Add the batch axis
    return digit_image


def process_and_predict_digits(crnn_model, directory_path):
    file_list = os.listdir(directory_path)

    for file_name in file_list:
        image_path = os.path.join(directory_path, file_name)
        img = cv2.imread(image_path)
        img = adfns.invert_thresh(img)
        print(img.shape)
        adfns.show_img(img, 3)

        preprocess_image = preprocess_image(img)

        adfns.show_img(preprocess_image, 3, title="After preprocessing")

        digit_images = extract_digits(preprocess_image)
        predictions = []

        for digit_image in digit_images:
            preprocessed_digit = preprocess_digit(digit_image)
            prediction = crnn_model.predict(preprocessed_digit)
            digit_prediction = np.argmax(prediction)
            predictions.append(digit_prediction)

        print("Predicted number:", "".join(map(str, predictions)))
        print(predictions)


# def extract_digits_bow(processed_image):
#     # Invert the binary image (black digits on white background)
#     inverted_image = cv2.bitwise_not(processed_image)

#     # Find contours in the inverted image
#     contours, _ = cv2.findContours(
#         inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#     )

#     # Initialize a list to store the bounding rectangles of each digit
#     digit_rects = []

#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         if w * h > 100:  # Filter out small/noisy contours
#             digit_rects.append((x, y, w, h))

#     # Sort the digit rectangles by their x-coordinate (left-to-right)
#     digit_rects = sorted(digit_rects, key=lambda x: x[0])

#     # Extract individual digits using the bounding rectangles
#     digit_images = [processed_image[y : y + h, x : x + w] for x, y, w, h in digit_rects]

#     return digit_images


# import cv2

# def extract_digits_bow(processed_image):
#     # Invert the binary image (black digits on white background)
#     inverted_image = cv2.bitwise_not(processed_image)

#     # Find contours in the inverted image
#     contours, _ = cv2.findContours(
#         inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#     )

#     # Initialize a list to store the bounding rectangles of each digit
#     digit_rects = []

#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         if w * h > 100:  # Filter out small/noisy contours
#             # Filter out decimal points by checking the aspect ratio
#             aspect_ratio = float(h) / w
#             if (
#                 aspect_ratio > 1.5
#             ):  # You can adjust this value based on your specific case
#                 digit_rects.append((x, y, w, h))

#     # Sort the digit rectangles by their x-coordinate (left-to-right)
#     digit_rects = sorted(digit_rects, key=lambda x: x[0])

#     # Extract individual digits using the bounding rectangles
#     digit_images = [processed_image[y : y + h, x : x + w] for x, y, w, h in digit_rects]

#     return digit_images


# Black digits on white background
def extract_digits_bow(processed_image):
    # Invert the binary image (black digits on white background)
    inverted_image = cv2.bitwise_not(processed_image)

    # Find contours in the inverted image
    contours, _ = cv2.findContours(
        inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Initialize a list to store the bounding rectangles of each digit
    digit_rects = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 100:  # Filter out small/noisy contours
            digit_rects.append((x, y, w, h))

    # Calculate the average height of the bounding rectangles
    average_height = sum([h for _, _, _, h in digit_rects]) / len(digit_rects)

    # Filter out decimal points by comparing each rectangle's height to the average height
    filtered_digit_rects = [
        rect for rect in digit_rects if rect[3] > 0.5 * average_height
    ]  # You can adjust this value based on your specific case

    # Sort the digit rectangles by their x-coordinate (left-to-right)
    filtered_digit_rects = sorted(filtered_digit_rects, key=lambda x: x[0])

    # Extract individual digits using the bounding rectangles
    digit_images = [
        processed_image[y : y + h, x : x + w] for x, y, w, h in filtered_digit_rects
    ]

    return digit_images
