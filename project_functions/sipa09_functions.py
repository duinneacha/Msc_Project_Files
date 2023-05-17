import cv2
import sys
import numpy as np
from PIL import Image as PILImage
from io import BytesIO

sys.path.append("../../Seperate_Folders/functions/")
import ad_functions as adfns
import sipa_rep_functions as sf
import ad_crnn_functions as adcrn

from tensorflow.keras.models import load_model
# model_file = "./models/CRNN_model.h5"
model_file = "./models/CRNN_digits_model_bw_600k_extra_layers.h5"
crnn_model = load_model(model_file)


def get_sipa09_images(file_list_segment, file_path):

    # print("in get_images")
    # Initialize an empty 5D NumPy array with dimensions (0, 7, 300, 300, 3)
    images_matrix = np.empty((0, 5, 300, 300, 3), dtype=np.uint8)
    digits_matrix = np.empty((0, 20), dtype=np.object)

    for index, filename in enumerate(file_list_segment):

        # print("filename:", filename)

        # Read the image and perform initial processing
        img = cv2.imread(file_path + filename)

        img_resized = adfns.read_resize_data(img, 80)

        gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # Perform deconvolution using the Wiener filter
        kernel_size = 5
        kernel_sd = 0
        signal_to_noise_ratio = 0.1
        deblurred = adfns.process_deblurred_image(
            gray_img, kernel_size, kernel_sd, signal_to_noise_ratio
        )

        # Apply thresholding to the gray image
        thresh = cv2.threshold(gray_img, 0, 55, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # print("thresh.shape:", thresh.shape)
        deblurred = cv2.cvtColor(deblurred, cv2.COLOR_GRAY2BGR)
        # gray_img = cv2.cvtColor(deblurred, cv2.COLOR_GRAY2BGR)

        # Inverted Thresholding
        inverted_thresh_img = adfns.invert_thresh(img_resized)
        # Read the dot matrix image from the masked image
        dot_matrix_image = adfns.read_dot_matrix_image(img_resized)

        (
            text_lets_original,
            text_ssd_original,
            text_eng_original,
            text_dotmatrix_original,
        ) = adfns.get_text_lets(img_resized)
        (
            text_lets_deblurred,
            text_ssd_deblurred,
            text_eng_deblurred,
            text_dotmatrix_deblurred,
        ) = adfns.get_text_lets(deblurred)
        (
            text_lets_thresh,
            text_ssd_thresh,
            text_eng_thresh,
            text_dotmatrix_thresh,
        ) = adfns.get_text_lets(thresh)
        (
            text_lets_dotmatrix,
            text_ssd_dotmatrix,
            text_eng_dotmatrix,
            text_dotmatrix_dotmatrix,
        ) = adfns.get_text_lets(dot_matrix_image)
        (
            text_lets_inverted_thresh,
            text_ssd_inverted_thresh,
            text_eng_inverted_thresh,
            text_dotmatrix_inverted_thresh,
        ) = adfns.get_text_lets(inverted_thresh_img)

        # print("text_dotmatrix_dotmatrix:", text_dotmatrix_dotmatrix)

        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        dot_matrix_image = cv2.cvtColor(dot_matrix_image, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inverted_thresh_img = cv2.cvtColor(inverted_thresh_img, cv2.COLOR_GRAY2BGR)

        # create an empty numpy array with the defined data types
        stacked_digits_array = np.empty((1, 20), dtype=np.object)

        # populate the stacked_digits_array with the values from the tesseract output
        stacked_digits_array[0, 0] = text_lets_original
        stacked_digits_array[0, 1] = text_ssd_original
        stacked_digits_array[0, 2] = text_eng_original
        stacked_digits_array[0, 3] = text_dotmatrix_original

        stacked_digits_array[0, 4] = text_lets_deblurred
        stacked_digits_array[0, 5] = text_ssd_deblurred
        stacked_digits_array[0, 6] = text_eng_deblurred
        stacked_digits_array[0, 7] = text_dotmatrix_deblurred

        stacked_digits_array[0, 8] = text_lets_thresh
        stacked_digits_array[0, 9] = text_ssd_thresh
        stacked_digits_array[0, 10] = text_eng_thresh
        stacked_digits_array[0, 11] = text_dotmatrix_thresh

        stacked_digits_array[0, 12] = text_lets_dotmatrix
        stacked_digits_array[0, 13] = text_ssd_dotmatrix
        stacked_digits_array[0, 14] = text_eng_dotmatrix
        stacked_digits_array[0, 15] = text_dotmatrix_dotmatrix

        stacked_digits_array[0, 16] = text_lets_inverted_thresh
        stacked_digits_array[0, 17] = text_ssd_inverted_thresh
        stacked_digits_array[0, 18] = text_eng_inverted_thresh
        stacked_digits_array[0, 19] = text_dotmatrix_inverted_thresh

        # print("stacked_digits_array[0, 27]", stacked_digits_array[0, 27])

        # Create a 5D NumPy array with dimensions (1, 7, 300, 300, 3) and copy the color channels
        images_line = np.empty((1, 5, 300, 300, 3), dtype=np.uint8)
        images_line[0, 0] = cv2.resize(img_rgb, (300, 300))
        images_line[0, 1] = cv2.resize(deblurred, (300, 300))
        images_line[0, 2] = cv2.resize(thresh, (300, 300))
        images_line[0, 3] = cv2.resize(dot_matrix_image, (300, 300))
        images_line[0, 4] = cv2.resize(inverted_thresh_img, (300, 300))

        # Add the images_line to the images_matrix array
        images_matrix = np.vstack((images_matrix, images_line))
        digits_matrix = np.vstack((digits_matrix, stacked_digits_array))

    return images_matrix, digits_matrix

def get_loaded_model():
    global crnn_model
    if crnn_model is None:
        crnn_model = load_model(model_file)
    return crnn_model


def get_sipa09_crnn_images(file_list_segment, file_path):


    # Load the CRNN model
    crnn_model = get_loaded_model()

    # Initialize an empty 5D NumPy array with dimensions (0, 7, 300, 300, 3)
    images_matrix = np.empty((0, 8, 300, 300, 3), dtype=np.uint8)
    digits_matrix = np.empty((0, 32), dtype=np.object)

    for index, filename in enumerate(file_list_segment):

        # Read the image and perform initial processing
        img = cv2.imread(file_path + filename)

        # Inverted Thresholding
        inverted_thresh_img = adfns.invert_thresh(img)
        (
            text_lets_original,
            text_ssd_original,
            text_eng_original,
            text_dotmatrix_original,
        ) = adfns.get_text_lets(img)
        (
            text_lets_inverted_thresh,
            text_ssd_inverted_thresh,
            text_eng_inverted_thresh,
            text_dotmatrix_inverted_thresh,
        ) = adfns.get_text_lets(inverted_thresh_img)
        
        preprocess_image = adcrn.preprocess_image(inverted_thresh_img)
        inverted_image = cv2.bitwise_not(preprocess_image)

        digit_images = adcrn.extract_digits_bow(inverted_image)
        predictions = []

        bgr_digit_images = []


        for digit_image in digit_images:
            preprocessed_digit = adcrn.preprocess_digit(digit_image)
            prediction = crnn_model.predict(preprocessed_digit)
            digit_prediction = np.argmax(prediction)
            # print("Predicted digit:", digit_prediction)
            predictions.append(digit_prediction)
            # print("digit_image.shape", digit_image.shape)
            bgr_digit_image = cv2.cvtColor(digit_image, cv2.COLOR_GRAY2BGR)
            bgr_digit_images.append(bgr_digit_image)
            # adfns.show_img(bgr_digit_image, size=1, title="bgr_digit_image")

        # Convert the list of BGR images to a NumPy array
        # bgr_digit_images_array = np.array(bgr_digit_images, dtype=object)
        # print("predictions", predictions)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inverted_thresh_img = cv2.cvtColor(inverted_thresh_img, cv2.COLOR_GRAY2BGR)

        

        images_line = np.empty((1, 8, 300, 300, 3), dtype=np.uint8)
        images_line[0, 0] = cv2.resize(img_rgb, (300, 300))
        images_line[0, 1] = cv2.resize(inverted_thresh_img, (300, 300))
        images_line[0, 2] = cv2.resize(bgr_digit_images[0], (300, 300))
        images_line[0, 3] = cv2.resize(bgr_digit_images[1], (300, 300))
        images_line[0, 4] = cv2.resize(bgr_digit_images[2], (300, 300))
        images_line[0, 5] = cv2.resize(bgr_digit_images[3], (300, 300))
        if len(bgr_digit_images) >= 5:
            images_line[0, 6] = cv2.resize(bgr_digit_images[4], (300, 300))
        else:
            # Handle the case when there is no element at index 4.
            # You can either assign a default image or leave it empty.
            pass
        if len(bgr_digit_images) >= 6:
            images_line[0, 7] = cv2.resize(bgr_digit_images[5], (300, 300))
        else:
            # Handle the case when there is no element at index 5.
            # You can either assign a default image or leave it empty.
            pass

        # images_line[0, 7] = cv2.resize(bgr_digit_images[5], (300, 300))
        # images_line[0, 2] = cv2.resize(inverted_thresh_img, (300, 300))

        # create an empty numpy array with the defined data types
        stacked_digits_array = np.empty((1, 32), dtype=np.object)

        stacked_digits_array[0,  0] = text_lets_original
        stacked_digits_array[0,  1] = text_ssd_original
        stacked_digits_array[0,  2] = text_eng_original
        stacked_digits_array[0,  3] = text_dotmatrix_original

        stacked_digits_array[0,  4] = text_lets_inverted_thresh
        stacked_digits_array[0,  5] = text_ssd_inverted_thresh
        stacked_digits_array[0,  6] = text_eng_inverted_thresh
        stacked_digits_array[0,  7] = text_dotmatrix_inverted_thresh

        stacked_digits_array[0,  8] = predictions[0]
        stacked_digits_array[0,  9] = ""
        stacked_digits_array[0, 10] = ""
        stacked_digits_array[0, 11] = ""
    
        stacked_digits_array[0, 12] = predictions[1]
        stacked_digits_array[0, 13] = ""
        stacked_digits_array[0, 14] = ""
        stacked_digits_array[0, 15] = ""
    
        stacked_digits_array[0, 16] = predictions[2]
        stacked_digits_array[0, 17] = ""
        stacked_digits_array[0, 18] = ""
        stacked_digits_array[0, 19] = ""
    
        stacked_digits_array[0, 20] = predictions[3]
        stacked_digits_array[0, 21] = ""
        stacked_digits_array[0, 22] = ""
        stacked_digits_array[0, 23] = ""
    
        if len(bgr_digit_images) >= 5:
            stacked_digits_array[0, 24] = predictions[4]
            stacked_digits_array[0, 25] = ""
            stacked_digits_array[0, 26] = ""
            stacked_digits_array[0, 27] = ""
        else:
            # Handle the case when there is no element at index 4.
            # You can either assign a default image or leave it empty.
            pass

        if len(bgr_digit_images) >= 6:
             stacked_digits_array[0, 28] = predictions[5]
             stacked_digits_array[0, 29] = ""
             stacked_digits_array[0, 30] = ""
             stacked_digits_array[0, 31] = ""
        else:
            # Handle the case when there is no element at index 5.
            # You can either assign a default image or leave it empty.
            pass

    
        # Add the images_line to the images_matrix array
        images_matrix = np.vstack((images_matrix, images_line))
        digits_matrix = np.vstack((digits_matrix, stacked_digits_array))

    return images_matrix, digits_matrix
    