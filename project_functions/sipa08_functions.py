import cv2
import sys
import numpy as np
from PIL import Image as PILImage
from io import BytesIO

sys.path.append("../../Seperate_Folders/functions/")
import ad_functions as adfns
import sipa_rep_functions as sf


# ###############################################################################################################################################
def get_images(file_list_segment, file_path):

    # print("in get_images")
    # Initialize an empty 5D NumPy array with dimensions (0, 7, 300, 300, 3)
    images_matrix = np.empty((0, 7, 300, 300, 3), dtype=np.uint8)
    digits_matrix = np.empty((0, 28), dtype=np.object)

    for index, filename in enumerate(file_list_segment):

        # print("filename:", filename)

        # Read the image and perform initial processing
        img = cv2.imread(file_path + filename)
        img_masked_green = adfns.mask_green(img)
        gray = cv2.cvtColor(img_masked_green, cv2.COLOR_BGR2GRAY)
        # print("gray.shape:", gray.shape)

        # Perform deconvolution using the Wiener filter
        kernel_size = 5
        kernel_sd = 0
        signal_to_noise_ratio = 0.1
        deblurred = adfns.process_deblurred_image(
            gray, kernel_size, kernel_sd, signal_to_noise_ratio
        )
        # print("deblurred.shape:", deblurred.shape)

        # Apply thresholding to the deblurred image
        thresh = cv2.threshold(deblurred, 0, 55, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # print("thresh.shape:", thresh.shape)
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        deblurred = cv2.cvtColor(deblurred, cv2.COLOR_GRAY2BGR)

        # Apply denoising to the masked image and color denoising to the masked image
        dst = cv2.fastNlMeansDenoisingColored(img_masked_green, None, 6, 16, 7, 21)
        denoise_image = adfns.denoise_image(img_masked_green)

        motsu_image = adfns.preprocess_motsu_image(img)
        # Read the dot matrix image from the masked image
        dot_matrix_image = adfns.read_dot_matrix_image(img_masked_green)
        # print("dot_matrix_image.shape:", dot_matrix_image.shape)
        dot_matrix_image = cv2.cvtColor(dot_matrix_image, cv2.COLOR_GRAY2BGR)
        motsu_image = adfns.preprocess_motsu_image(img)
        motsu_image = cv2.cvtColor(motsu_image, cv2.COLOR_GRAY2BGR)

        (
            text_lets_original,
            text_ssd_original,
            text_eng_original,
            text_dotmatrix_original,
        ) = adfns.get_text_lets(img)
        (
            text_lets_masked,
            text_ssd_masked,
            text_eng_masked,
            text_dotmatrix_masked,
        ) = adfns.get_text_lets(img_masked_green)
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
            text_lets_deionised,
            text_ssd_deionised,
            text_eng_deionised,
            text_dotmatrix_deionised,
        ) = adfns.get_text_lets(denoise_image)

        (
            text_lets_dst,
            text_ssd_dst,
            text_eng_dst,
            text_dotmatrix_dst,
        ) = adfns.get_text_lets(dst)

        (
            text_lets_dotmatrix,
            text_ssd_dotmatrix,
            text_eng_dotmatrix,
            text_dotmatrix_dotmatrix,
        ) = adfns.get_text_lets(dot_matrix_image)

        (
            text_lets_motsu,
            text_ssd_motsu,
            text_eng_motsu,
            text_dotmatrix_motsu,
        ) = adfns.get_text_lets(motsu_image)

        # print("text_dotmatrix_dotmatrix:", text_dotmatrix_dotmatrix)

        # create an empty numpy array with the defined data types
        stacked_digits_array = np.empty((1, 28), dtype=np.object)

        # populate the stacked_digits_array with the values from the tesseract output
        stacked_digits_array[0, 0] = text_lets_original
        stacked_digits_array[0, 1] = text_ssd_original
        stacked_digits_array[0, 2] = text_eng_original
        stacked_digits_array[0, 3] = text_dotmatrix_original

        stacked_digits_array[0, 4] = text_lets_masked
        stacked_digits_array[0, 5] = text_ssd_masked
        stacked_digits_array[0, 6] = text_eng_masked
        stacked_digits_array[0, 7] = text_dotmatrix_masked

        stacked_digits_array[0, 8] = text_lets_deblurred
        stacked_digits_array[0, 9] = text_ssd_deblurred
        stacked_digits_array[0, 10] = text_eng_deblurred
        stacked_digits_array[0, 11] = text_dotmatrix_deblurred

        stacked_digits_array[0, 12] = text_lets_thresh
        stacked_digits_array[0, 13] = text_ssd_thresh
        stacked_digits_array[0, 14] = text_eng_thresh
        stacked_digits_array[0, 15] = text_dotmatrix_thresh

        # stacked_digits_array[0, 16] = text_lets_deionised
        # stacked_digits_array[0, 17] = text_ssd_deionised
        # stacked_digits_array[0, 18] = text_eng_deionised
        # stacked_digits_array[0, 19] = text_dotmatrix_deionised
        stacked_digits_array[0, 16] = text_lets_motsu
        stacked_digits_array[0, 17] = text_ssd_motsu
        stacked_digits_array[0, 18] = text_eng_motsu
        stacked_digits_array[0, 19] = text_dotmatrix_motsu

        stacked_digits_array[0, 20] = text_lets_dst
        stacked_digits_array[0, 21] = text_ssd_dst
        stacked_digits_array[0, 22] = text_eng_dst
        stacked_digits_array[0, 23] = text_dotmatrix_dst

        stacked_digits_array[0, 24] = text_lets_dotmatrix
        stacked_digits_array[0, 25] = text_ssd_dotmatrix
        stacked_digits_array[0, 26] = text_eng_dotmatrix
        stacked_digits_array[0, 27] = text_dotmatrix_dotmatrix

        # print("stacked_digits_array[0, 27]", stacked_digits_array[0, 27])

        # Create a 5D NumPy array with dimensions (1, 7, 300, 300, 3) and copy the color channels
        images_line = np.empty((1, 7, 300, 300, 3), dtype=np.uint8)
        images_line[0, 0] = cv2.resize(img, (300, 300))
        images_line[0, 1] = cv2.resize(img_masked_green, (300, 300))
        images_line[0, 2] = cv2.resize(deblurred, (300, 300))
        images_line[0, 3] = cv2.resize(thresh, (300, 300))
        images_line[0, 4] = cv2.resize(motsu_image, (300, 300))
        # images_line[0, 5] = cv2.resize(denoise_image, (300, 300))
        images_line[0, 5] = cv2.resize(dst, (300, 300))
        images_line[0, 6] = cv2.resize(dot_matrix_image, (300, 300))

        # print(f"images_matrix shape before vstack: {images_matrix.shape}")
        # print(f"images_line shape: {images_line.shape}")

        # Add the images_line to the images_matrix array
        images_matrix = np.vstack((images_matrix, images_line))
        digits_matrix = np.vstack((digits_matrix, stacked_digits_array))

        # print(f"images_matrix shape after vstack: {images_matrix.shape}")

    return images_matrix, digits_matrix


# ###############################################################################################################################################
def get_two_images(file_list_segment, file_path):
    # print("in get_images")
    # Initialize an empty 5D NumPy array with dimensions (0, 7, 300, 300, 3)
    images_matrix = np.empty((0, 2, 300, 300, 3), dtype=np.uint8)
    digits_matrix = np.empty((0, 8), dtype=np.str)

    for index, filename in enumerate(file_list_segment):
        # Read the image and perform initial processing
        img = cv2.imread(file_path + filename)
        img_masked_green = adfns.mask_green(img)

        gray = cv2.cvtColor(img_masked_green, cv2.COLOR_BGR2GRAY)

        # # Perform deconvolution using the Wiener filter
        # kernel_size = 5
        # kernel_sd = 0
        # signal_to_noise_ratio = 0.1
        # deblurred = adfns.process_deblurred_image(
        #     gray, kernel_size, kernel_sd, signal_to_noise_ratio
        # )

        # # Apply thresholding to the deblurred image
        # thresh = cv2.threshold(deblurred, 0, 55, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        # deblurred = cv2.cvtColor(deblurred, cv2.COLOR_GRAY2BGR)

        # # Apply denoising to the masked image and color denoising to the masked image
        # dst = cv2.fastNlMeansDenoisingColored(img_masked_green, None, 6, 16, 7, 21)
        # denoise_image = adfns.denoise_image(img_masked_green)

        # # Read the dot matrix image from the masked image
        # dot_matrix_image = adfns.read_dot_matrix_image(img_masked_green)

        # dot_matrix_image = cv2.cvtColor(dot_matrix_image, cv2.COLOR_GRAY2BGR)

        (
            text_lets_original,
            text_ssd_original,
            text_eng_original,
            text_dotmatrix_original,
        ) = adfns.get_text_lets(img)
        (
            text_lets_masked,
            text_ssd_masked,
            text_eng_masked,
            text_dotmatrix_masked,
        ) = adfns.get_text_lets(img_masked_green)

        # # stack the four arrays
        stacked_digits_array = np.empty((1, 8), dtype=np.str)
        # stacked_array = np.empty((8,), dtype=np.float64)
        # print(stacked_array.shape)

        # # use slicing to populate the stacked array with the four original arrays and the four masked arrays
        stacked_digits_array[0, 1] = text_lets_original
        stacked_digits_array[0, 1] = text_ssd_original
        stacked_digits_array[0, 2] = text_eng_original
        stacked_digits_array[0, 3] = text_dotmatrix_original
        stacked_digits_array[0, 4] = text_lets_masked
        stacked_digits_array[0, 5] = text_ssd_masked
        stacked_digits_array[0, 6] = text_eng_masked
        stacked_digits_array[0, 7] = text_dotmatrix_masked

        # Create a 5D NumPy array with dimensions (1, 7, 300, 300, 3) and copy the color channels
        images_line = np.empty((1, 2, 300, 300, 3), dtype=np.uint8)
        images_line[0, 0] = cv2.resize(img, (300, 300))
        images_line[0, 1] = cv2.resize(img_masked_green, (300, 300))
        # images_line[0, 2] = cv2.resize(img_masked_green, (300, 300))

        # Add the images_line to the images_matrix array
        images_matrix = np.vstack((images_matrix, images_line))
        digits_matrix = np.vstack((digits_matrix, stacked_digits_array))

        # print(f"images_matrix shape after vstack: {images_matrix.shape}")

    return images_matrix, digits_matrix


# ###############################################################################################################################################
def format_image_for_pdf(raw_img):
    """
    Takes a NumPy array representing an image and converts it to a PNG image in memory.

    Parameters:
    raw_img (numpy.ndarray): A NumPy array representing the image to be converted.

    Returns:
    io.BytesIO: A BytesIO object containing the PNG image data.
    """

    # Convert the NumPy array to a PIL image object
    pil_img = PILImage.fromarray(raw_img.astype(np.uint8))

    # Create a BytesIO object to store the PNG image data
    img_data_png = BytesIO()

    # Save the PIL image object as a PNG to the BytesIO object
    pil_img.save(img_data_png, format="PNG")

    # Reset the BytesIO object to the beginning of the stream
    img_data_png.seek(0)

    # Return the BytesIO object containing the PNG image data
    return img_data_png


def parse_digits_list(digits_list):
    print("in parse_digits_list")
    print(f"digits_list: {digits_list}")

    return "AD"
