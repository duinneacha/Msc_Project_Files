from matplotlib import pyplot as plt
import pytesseract
import pandas as pd
import numpy as np
import random
import cv2
import os
import re


#########################################################################################################
# Function to show an image using pyplot
def show_img(img, size=12, title=None):
    """
    Displays an image using Matplotlib.

    Args:
        img (numpy.ndarray): The image to display, in BGR format.
        size (int, optional): The size of the displayed figure. Default is 12.
        title (str, optional): The title to display above the image. Default is None.

    Returns:
        None
    """

    # Get the current figure and set its size to the given size.
    fig = plt.gcf()
    fig.set_size_inches(size, size)

    # Turn off the axis to remove the ticks and labels.
    plt.axis("off")

    # Convert the image from BGR to RGB and display it using Matplotlib.
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # If a title is provided, set the title of the plot.
    if title is not None:
        plt.title(title)

    # Show the plot.
    plt.show()


#########################################################################################################
# Display four images
def display_four_images(folder_path, title=None):
    """
      Displays four random images from a directory using Matplotlib.

    Args:
        folder_path (str): The path to the directory containing the images.
        title (str, optional): The title to display above the images. Default is None.

    Returns:
        None
    """
    # Get the list of all files in the directory
    filenames = os.listdir(folder_path)
    # Select 4 random files from the list
    random_files = random.sample(filenames, 4)

    # Initialize the figure and axis
    fig, axs = plt.subplots(2, 2, figsize=(5, 5), facecolor="#808080")

    # Iterate through the selected random files and plot the image
    for i, f in enumerate(random_files):
        img = cv2.imread(os.path.join(folder_path, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i // 2, i % 2].imshow(img)
        axs[i // 2, i % 2].axis("off")
        axs[i // 2, i % 2].grid(False)
        axs[i // 2, i % 2].set_title(f)

    # Remove the grid lines from the empty subplots
    for ax in axs.flat:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#808080")
        for sp in ax.spines.values():
            sp.set_visible(False)

    # Set the overall chart title
    fig.suptitle(title, fontsize=20, fontweight=700)

    plt.show()


#########################################################################################################
def display_ten_images(folder_path, title=None):
    """
    Displays ten images from a directory using Matplotlib.

    Args:
        folder_path (str): The path to the directory containing the images.
        title (str, optional): The title to display above the images. Default is None.

    Returns:
        None
    """
    # Get the list of all files in the directory
    filenames = os.listdir(folder_path)
    # Select the first 10 files from the list
    selected_files = filenames[:10]

    # Initialize the figure and axis
    fig, axs = plt.subplots(5, 2, figsize=(10, 12), facecolor="#808080")

    # Iterate through the selected files and plot the image
    for i, f in enumerate(selected_files):
        img = cv2.imread(os.path.join(folder_path, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i // 2, i % 2].imshow(img)
        axs[i // 2, i % 2].axis("off")
        axs[i // 2, i % 2].grid(False)
        axs[i // 2, i % 2].set_title(f)

    # Remove the grid lines from the empty subplots
    for ax in axs.flat:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#808080")
        for sp in ax.spines.values():
            sp.set_visible(False)

    # Set the overall chart title
    fig.suptitle(title, fontsize=20, fontweight=700)

    plt.show()


#########################################################################################################
# Read Input File - parameter number of the folder
def read_input_file_list(folder_number):
    """
    Reads a list of input image files from a text file and filters them based on a folder number.

    Args:
        folder_number (int or str): The folder number to filter by.

    Returns:
        A pandas DataFrame containing the filtered input file list.
    """

    # Read image file list
    input_data = pd.read_csv(
        r"./labelled_images_input_no_masks.txt",
        names=["file_name", "seen_value", "ncol2", "ncol3"],
        sep="\t",
        header=None,
    )
    input_data = input_data.reset_index()

    # Filter based on the number that lies between the second and third backslashes - using preceeded and followed by RE
    filtered_data = input_data[
        input_data["file_name"].str.contains(
            re.compile(r"\\{}\\".format(folder_number))
        )
    ]

    print(len(filtered_data), "input images to process!!")

    return filtered_data


#########################################################################################################
def is_image(image):
    """
    Check if an input is a valid image array.

    Args:
        image (numpy.ndarray): The input array to check.

    Returns:
        bool: True if the input is a valid image array, False otherwise.
    """
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2 or (
            len(image.shape) == 3 and image.shape[2] in [1, 3]
        ):
            return True
    return False


#########################################################################################################
# Read image file and size as per parameter and return the resized image
def read_resize_data(input_file, size=160):
    """
    Reads an input image file, resizes it to a specified size, and returns the resized image.

    Args:
        input_file (str): The path to the input image file.
        size (int, optional): The desired size of the output image. Default is 160.

    Returns:
        A resized image as a numpy.ndarray.
    """

    if is_image(input_file):
        img = input_file
    else:
        # Read image
        img = cv2.imread(input_file)

    # Set dimensions
    width = size
    height = size
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized


#########################################################################################################
# Read text using pytesseract and return the text
def get_text(image):
    """
    Uses Tesseract OCR to extract text from an input image.

    Args:
        image (numpy.ndarray): The input image to extract text from.

    Returns:
        A tuple containing the extracted text as strings: (text_ssd, text_eng).
    """

    # psm 6 = Assume a single uniform block of text.
    config_tesseract = "--tessdata-dir ./ttesseract_langs --psm 6"

    # Read text from image using Seven Segment training data
    # text_ssd = pytesseract.image_to_string(image, lang="ssd", config=config_tesseract)
    text_ssd = pytesseract.image_to_string(image, lang="ssd", config=config_tesseract)

    # Read text from image using English character training data
    text_eng = pytesseract.image_to_string(image, lang="eng", config=config_tesseract)

    # Clean text
    text_ssd = "".join(c for c in text_ssd if c.isdigit() or c == ".")
    text_eng = "".join(c for c in text_eng if c.isdigit() or c == ".")

    return text_ssd, text_eng


#########################################################################################################
# Read text using pytesseract and return the text
def get_text_lets(image):
    """
    Uses Tesseract OCR to extract text from an input image.

    Args:
        image (numpy.ndarray): The input image to extract text from.

    Returns:
        A tuple containing the extracted text as strings: (text_lets, text_eng).
    """

    # psm 6 = Assume a single uniform block of text.
    config_tesseract = "--tessdata-dir ./ttesseract_langs --oem 1 --psm 7"

    # Read text from image using Seven Segment training data

    # text_ssd = pytesseract.image_to_string(image, lang="letsgodigital", config=config_tesseract)
    text_ssd = pytesseract.image_to_string(image, lang="ssd", config=config_tesseract)

    # text_lets = pytesseract.image_to_string(image, lang="ssd", config=config_tesseract)
    text_lets = pytesseract.image_to_string(
        image, lang="letsgodigital", config=config_tesseract
    )

    # Read text from image using English character training data
    text_eng = pytesseract.image_to_string(image, lang="eng", config=config_tesseract)

    text_dmx = pytesseract.image_to_string(
        image, lang="dotslayer", config=config_tesseract
    )

    # Clean text
    text_lets = "".join(c for c in text_lets if c.isdigit() or c == ".")
    text_ssd = "".join(c for c in text_ssd if c.isdigit() or c == ".")
    text_eng = "".join(c for c in text_eng if c.isdigit() or c == ".")
    text_dmx = "".join(c for c in text_dmx if c.isdigit() or c == ".")

    return text_lets, text_ssd, text_eng, text_dmx


#########################################################################################################
# Process using Thresh and return text for Seven Segment Display and English
def process_file_thresh(input_file, size=None):
    """
    Processes an input image file by resizing it, converting it to grayscale, and applying binary thresholding.
    Extracts text using Tesseract OCR with both Seven Segment and English character training data.

    Args:
        input_file (str): The path to the input image file.
        size (int, optional): The desired size of the output image after resizing. If not provided, the original size is used.

    Returns:
        A tuple containing the extracted text as strings: (text_ssd, text_eng).
    """

    # Get image data
    # img = read_input_data(input_file)
    if size:
        img = read_resize_data(input_file, size)
    else:
        img = cv2.imread(input_file)

    # Convert to RGB (three dimensions)
    nimRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to gray (one dimension)
    nimGray = cv2.cvtColor(nimRGB, cv2.COLOR_BGR2GRAY)

    # Set the minimum to gray and max to white
    min_threshold = 127
    max_threshold = 255

    # Binary Thresh
    value, nimThresh = cv2.threshold(
        nimGray, min_threshold, max_threshold, cv2.THRESH_BINARY
    )

    # Get Text for Seven Segment and English
    text_ssd, text_eng = get_text(nimThresh)

    return text_ssd, text_eng


#########################################################################################################
# Process using Closing and return text for Seven Segment Display and English
def process_file_closing(input_file, size=None):
    """
    Processes an input image file by resizing it, converting it to grayscale, and applying a closing operation
    using dilation and erosion with 5x5 matrices. Extracts text using Tesseract OCR with both Seven Segment and English
    character training data.

    Args:
        input_file (str): The path to the input image file.
        size (int, optional): The desired size of the output image after resizing. If not provided, the original size is used.

    Returns:
        A tuple containing the extracted text as strings: (text_ssd, text_eng).
    """

    # Get image data
    # img = read_input_data(input_file)
    if size:
        img = read_resize_data(input_file, size)
    else:
        img = cv2.imread(input_file)

    # Convert to RGB (three dimensions)
    nimRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to gray (one dimension)
    nimGray = cv2.cvtColor(nimRGB, cv2.COLOR_BGR2GRAY)

    # Perform Dilation using a 5x5 matrix
    cdilation = cv2.dilate(nimGray, np.ones((5, 5), np.uint8))

    # PErform Erod using a 5x5 matrix
    nimClosing = cv2.erode(cdilation, np.ones((5, 5), np.uint8))

    # Get Text for Seven Segment and English
    text_ssd, text_eng = get_text(nimClosing)

    return text_ssd, text_eng


#########################################################################################################
# Process using Adaptive Gaussian and return text for Seven Segment Display and English
def process_adaptive_gaussian(input_file, size=None):
    """
    Processes an input image file by resizing it, converting it to grayscale, and applying adaptive thresholding
    using a Gaussian-weighted sum of the neighbourhood pixel values. Extracts text using Tesseract OCR with both
    Seven Segment and English character training data.

    Args:
        input_file (str): The path to the input image file.
        size (int, optional): The desired size of the output image after resizing. If not provided, the original size is used.

    Returns:
        A tuple containing the extracted text as strings: (text_ssd, text_eng).
    """

    # Get image data
    # img = read_input_data(input_file)
    if size:
        img = read_resize_data(input_file, size)
    else:
        img = cv2.imread(input_file)

    # Convert to RGB (three dimensions)
    nimRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to gray (one dimension)
    nimGray = cv2.cvtColor(nimRGB, cv2.COLOR_BGR2GRAY)

    # 255 is a value that is going to be assigned to respectively pixels in the result
    # (namely, to all pixels which value in the source is greater then computed threshold level)
    max_threshold = 255

    adaptive_gaussian = cv2.adaptiveThreshold(
        nimGray, max_threshold, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9
    )

    # Get Text for Seven Segment and English
    text_ssd, text_eng = get_text(adaptive_gaussian)

    return text_ssd, text_eng


#########################################################################################################
# Process using Closing and return text for Seven Segment Display and English
def process_otsu(input_file, size=None):
    """
    Processes an input image file by resizing it, converting it to grayscale, and applying Otsu's thresholding method
    for image segmentation. Extracts text using Tesseract OCR with both Seven Segment and English character training
    data.

    Args:
        input_file (str): The path to the input image file.
        size (int, optional): The desired size of the output image after resizing. If not provided, the original size is used.

    Returns:
        A tuple containing the extracted text as strings: (text_ssd, text_eng).
    """

    # Get image data
    if size:
        img = read_resize_data(input_file, size)
    else:
        img = cv2.imread(input_file)

    # Convert to RGB (three dimensions)
    nimRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to gray (one dimension)
    nimGray = cv2.cvtColor(nimRGB, cv2.COLOR_BGR2GRAY)

    # 0 means threshold level which actually is omitted because we used CV_THRESH_OTSU flag
    min_threshold = 0

    # 255 is a value that is going to be assigned to respectively pixels in the result
    # (namely, to all pixels which value in the source is greater then computed threshold level)
    max_threshold = 255

    # THRESH_BINARY | THRESH_OTSU is a required flag to perform Otsu thresholding. Because in fact we would like to perform binary thresholding,
    # so we use CV_THRESH_BINARY (you can use any of 5 flags opencv provides) combined with CV_THRESH_OTSU
    value, nimOTSU = cv2.threshold(
        nimGray, min_threshold, max_threshold, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    # Get Text for Seven Segment and English
    text_ssd, text_eng = get_text(nimOTSU)

    return text_ssd, text_eng


#########################################################################################################
def read_dot_matrix_image(image):
    """
    Takes an image and performs image processing operations to create a binary image with black dots on a white background,
    which represents the original dot matrix image.

    Args:
    - image: a numpy.ndarray representing the input image

    Returns:
    - close: a numpy.ndarray representing the processed image
    """

    # Load the image using OpenCV
    # image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Perform morphological operations to remove noise and fill in gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Return the processed image
    return close


#########################################################################################################
# def process_deblurred_image(gray_image, kernel_size, kernel_sd, signal_to_noise_ratio):

#     # Define the blur kernel (e.g., Gaussian blur kernel)
#     kernel = cv2.getGaussianKernel(kernel_size, kernel_sd)
#     psf = kernel @ kernel.T

#     # Define the Wiener filter

#     psf /= np.sum(psf)
#     otf = np.fft.fft2(psf)
#     otf = np.conj(otf) / (np.abs(otf) ** 2 + signal_to_noise_ratio)
#     wiener_filter = np.fft.ifft2(otf)

#     # Perform deconvolution using the Wiener filter
#     deblurred = cv2.filter2D(gray_image, -1, wiener_filter.real)

#     return deblurred


def process_deblurred_image(gray_image, kernel_size, kernel_sd, signal_to_noise_ratio):
    """
    Applies deconvolution to a grayscale image using a Wiener filter with a Gaussian blur kernel.

    Args:
        gray_image (numpy.ndarray): The grayscale input image as a 2D NumPy array.
        kernel_size (int): The size of the Gaussian blur kernel.
        kernel_sd (float): The standard deviation of the Gaussian blur kernel.
        signal_to_noise_ratio (float): The signal-to-noise ratio used for the Wiener filter.

    Returns:
        numpy.ndarray: The deblurred grayscale image as a 2D NumPy array.
    """

    # Define the blur kernel (e.g., Gaussian blur kernel)
    kernel = cv2.getGaussianKernel(kernel_size, kernel_sd)
    psf = kernel @ kernel.T

    # Define the Wiener filter
    psf /= np.sum(psf)
    otf = np.fft.fft2(psf)
    otf = np.conj(otf) / (np.abs(otf) ** 2 + signal_to_noise_ratio)
    wiener_filter = np.fft.ifft2(otf)

    # Perform deconvolution using the Wiener filter
    deblurred = cv2.filter2D(gray_image, -1, wiener_filter.real)

    return deblurred


#########################################################################################################
# Control loop for Sipa3 using stated size calculate thresh, closing, otsu and return array
def control_loop_sipa3(input_data, size_to_process):

    """
    Processes a list of input image files by applying several image processing techniques and extracting text using
    Tesseract OCR with both Seven Segment and English character training data. Returns a list of arrays containing the
    processed data for each input image.

    Args:
        input_data (pandas.DataFrame): A pandas DataFrame containing the list of input image files and their corresponding
        metadata.
        size_to_process (int): The desired size of the output images after resizing.

    Returns:
        A list of arrays containing the processed data for each input image. Each array contains the following data:
        [input_file, seen_value, size_to_process, thresh_ssd, thresh_eng, closing_ssd, closing_eng, otsu_ssd,
        otsu_eng, gaus_ssd, gaus_eng].
    """

    input_array = []
    # print(size_to_process)

    for row in input_data.iterrows():

        input_file = row[1][1]
        seen_value = row[1][2]
        # print("input_file", input_file)

        # Thresh
        thresh_ssd, thresh_eng = process_file_thresh(input_file, size_to_process)

        # Dialation
        closing_ssd, closing_eng = process_file_closing(input_file, size_to_process)

        # OTSU
        otsu_ssd, otsu_eng = process_otsu(input_file, size_to_process)

        # Gaussian
        gaus_ssd, gaus_eng = process_adaptive_gaussian(input_file, size_to_process)

        # Add text to array
        new_row = [
            input_file,
            seen_value,
            size_to_process,
            thresh_ssd,
            thresh_eng,
            closing_ssd,
            closing_eng,
            otsu_ssd,
            otsu_eng,
            gaus_ssd,
            gaus_eng,
        ]

        input_array.append(new_row)

    return input_array


#########################################################################################################
def crop_image(img, top=0, bottom=0, left=0, right=0):
    """
    Crop an image on all sides, only if there is a value for top, bottom, right, and left.

    Parameters:
        img (numpy.ndarray): The image to be cropped.
        top (int, optional): The number of pixels to be cropped from the top. Default is 0.
        bottom (int, optional): The number of pixels to be cropped from the bottom. Default is 0.
        left (int, optional): The number of pixels to be cropped from the left. Default is 0.
        right (int, optional): The number of pixels to be cropped from the right. Default is 0.

    Returns:
        numpy.ndarray: The cropped image.
    """
    # Check if there is a value for each side
    if top:
        img = img[top:, :]
    if bottom:
        img = img[:-bottom, :]
    if left:
        img = img[:, left:]
    if right:
        img = img[:, :-right]

    return img


#########################################################################################################
def denoise_image(
    image, h=6, hForColorComponents=6, templateWindowSize=7, searchWindowSize=21
):
    """
    Apply non-local means denoising to an input image using OpenCV.

    Args:
        image (numpy.ndarray): A 3-channel color image with pixel values in the range [0, 255].
        h (float): The parameter that controls the filter strength. A larger value of `h` results in stronger denoising. Default is 6.
        hForColorComponents (float): The parameter that controls the filter strength for color images. A larger value of `hForColorComponents` results in stronger denoising of color components. Default is 6.
        templateWindowSize (int): The size of the window used for the non-local means algorithm. A larger value of `templateWindowSize` results in smoother denoising, but at the cost of blurring fine details. Default is 7.
        searchWindowSize (int): The size of the search window used for the non-local means algorithm. A larger value of `searchWindowSize` results in better noise reduction, but at the cost of longer computation time. Default is 21.

    Returns:
        numpy.ndarray: The denoised image.

    """
    return cv2.fastNlMeansDenoisingColored(
        image, None, h, hForColorComponents, templateWindowSize, searchWindowSize
    )


#########################################################################################################
def mask_red(img):
    """
    Returns an image with only the red pixels from the input image.

    Parameters:
    img (numpy.ndarray): Input image in BGR format.

    Returns:
    numpy.ndarray: Output image with only the red pixels from the input image.

    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define lower and upper red ranges in HSV
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # create masks with the specified red ranges
    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)

    # join the masks
    mask = mask1 + mask2

    # set the output image to zero everywhere except the red mask
    output_img = img.copy()
    output_img[np.where(mask == 0)] = 0

    return output_img


#########################################################################################################
def mask_green(img):
    """
    Returns an image with only the green pixels from the input image.

    Parameters:
    img (numpy.ndarray): Input image in BGR format.

    Returns:
    numpy.ndarray: Output image with only the green pixels from the input image.

    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of green color in HSV
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    # create a mask with the specified green range
    mask = cv2.inRange(img_hsv, lower_green, upper_green)

    # apply the mask to the image to get the output image
    output_img = cv2.bitwise_and(img, img, mask=mask)

    return output_img


#########################################################################################################


# def preprocess_motsu_image(img):

#     masked_green = mask_green(img)
#     masked_green_gray = cv2.cvtColor(masked_green, cv2.COLOR_BGR2GRAY)
#     blur_kernel = (3, 3)
#     blurred_img = cv2.GaussianBlur(masked_green_gray, blur_kernel, 0)
#     inverted_img = cv2.bitwise_not(blurred_img)
#     thresh_img = cv2.threshold(
#         inverted_img, 80, 155, cv2.THRESH_BINARY | cv2.THRESH_OTSU
#     )[1]
#     blur_img = cv2.GaussianBlur(thresh_img, (5, 5), 0)
#     ret, motsu_img = cv2.threshold(
#         blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
#     )

#     return motsu_img


#########################################################################################################
def preprocess_motsu_image(img):
    """
    Preprocess an input image using the Motsu thresholding technique.
    This function applies a series of image processing techniques, including
    green color masking, grayscale conversion, Gaussian blurring, inversion,
    and thresholding. The resulting image is suitable for further processing or analysis.

    Args:
        img (numpy.ndarray): A BGR input image (3-channel).

    Returns:
        numpy.ndarray: A preprocessed binary image using Motsu thresholding.
    """

    # Mask the green color in the input image
    masked_green = mask_green(img)

    # Convert the masked image to grayscale
    masked_green_gray = cv2.cvtColor(masked_green, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur with a kernel size of (3, 3)
    blur_kernel = (5, 5)
    blurred_img = cv2.GaussianBlur(masked_green_gray, blur_kernel, 0)

    # Invert the blurred image
    inverted_img = cv2.bitwise_not(blurred_img)

    # Apply Motsu thresholding with a binary threshold and Otsu's method
    thresh_img = cv2.threshold(
        inverted_img, 80, 155, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]

    # Apply a second Gaussian blur with a kernel size of (5, 5)
    blur_img = cv2.GaussianBlur(thresh_img, (5, 5), 0)

    # Apply the final Motsu thresholding
    ret, motsu_img = cv2.threshold(
        blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return motsu_img


#########################################################################################################
def delete_file(filename):
    """
    Delete the specified file if it exists.

    Parameters:
    filename (str): The name of the file to delete.

    Returns:
    None.
    """
    if os.path.exists(filename):  # check if the file exists
        os.remove(filename)  # remove the file
        print(f"{filename} has been deleted.")
    else:
        print(f"{filename} does not exist.")  # print message if the file does not exist


#########################################################################################################
def invert_thresh(img):
    """
    Invert the image colors and apply a binary threshold using Otsu's method.

    This function first inverts the colors of the input image, then converts it
    to grayscale, and finally applies a binary threshold using Otsu's method.
    The result is a binary image with inverted colors and thresholding.

    Args:
        img (numpy.ndarray): The input image in BGR format.

    Returns:
        numpy.ndarray: The inverted and thresholded binary image.
    """
    # Invert the colors of the input image
    inverted_img = cv2.bitwise_not(img)

    # Convert the inverted image to grayscale
    gray_img = cv2.cvtColor(inverted_img, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold using Otsu's method
    invert_thresh_img = cv2.threshold(
        gray_img, 80, 155, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]

    # Return the inverted and thresholded binary image
    return invert_thresh_img


#########################################################################################################
def plot_label_frequencies(df, label_column, df_name):
    """
    Plot a horizontal bar chart of label frequencies in a DataFrame.

    This function takes a DataFrame, a label column name, and the DataFrame name as input,
    groups the labels into frequency bins, calculates the percentage of labels in each bin,
    and visualizes the distribution using a horizontal bar chart with the 'viridis' color scheme.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the label data.
    label_column : str
        The name of the column containing the labels in the DataFrame.
    df_name : str
        The name of the DataFrame, used in the plot title.

    Returns
    -------
    None
    """

    # Calculate the quantity of each label
    label_counts = df[label_column].value_counts()

    # Define frequency bins
    bins = pd.cut(
        label_counts, bins=[0, 1, 5, 10, 50, 100, 500, 1000, float("inf")], right=False
    )

    # Group label counts by frequency bins
    binned_counts = label_counts.groupby(bins).count()

    # Calculate the percentage of labels in each bin
    binned_percentage = binned_counts / binned_counts.sum() * 100

    # Set the 'viridis' color scheme
    colors = plt.cm.viridis(np.linspace(0, 1, len(binned_counts)))

    # Plot the horizontal bar chart
    plt.figure(figsize=(12, 6))
    ax = binned_percentage.plot(kind="barh", color=colors)
    plt.xlabel("Percentage of Labels")
    plt.ylabel("Label Frequency Range")
    plt.title(f"Label Frequencies in {df_name}")

    # Add percentages next to bars
    for i, percentage in enumerate(binned_percentage):
        ax.text(percentage + 0.5, i, f"{percentage:.2f}%", va="center", fontsize=10)

    plt.show()


#########################################################################################################
def get_rotation_angle(image):
    """
    Estimate the rotation angle of an image by detecting lines using the Hough Transform.

    :param image: Input image, assumed to be in BGR format
    :return: Estimated rotation angle in degrees; returns 0 if no lines are detected
    """
    # Convert the input image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the Canny edge detector to find edges in the image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines in the image using the Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    # If no lines are detected, return 0 as the rotation angle
    if lines is None:
        return 0

    # Calculate the angles of the detected lines
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta) - 90
        angles.append(angle)

    # Compute the median angle as the estimated rotation angle
    median_angle = np.median(angles)

    return median_angle


#########################################################################################################
def rotate_image(image, angle):
    """
    Rotate an image by a specified angle.

    :param image: Input image, assumed to be in BGR format
    :param angle: Angle in degrees to rotate the image, positive values mean counter-clockwise rotation
    :return: Rotated image
    """
    # Get the dimensions of the input image
    height, width = image.shape[:2]

    # Calculate the center of the image
    center = (width // 2, height // 2)

    # Compute the rotation matrix for the given angle
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply the rotation matrix to the input image using cv2.warpAffine()
    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return rotated_image
