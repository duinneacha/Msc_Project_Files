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


# def read_input_file_list(folder_number):

#     # Read image file list
#     input_data = pd.read_csv(r'./labelled_images_input_no_masks.txt', names=["file_name","seen_value","ncol2","ncol3"], sep="\t", header=None)
#     input_data = input_data.reset_index()

#     # Display Image Count
#     print(len(input_data), "input images to process!!")
#     return input_data


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
    text_ssd = pytesseract.image_to_string(image, lang="ssd", config=config_tesseract)

    # Read text from image using English character training data
    text_eng = pytesseract.image_to_string(image, lang="eng", config=config_tesseract)

    # Clean text
    text_ssd = "".join(c for c in text_ssd if c.isdigit() or c == ".")
    text_eng = "".join(c for c in text_eng if c.isdigit() or c == ".")

    return text_ssd, text_eng


#########################################################################################################
# Process using Thresh and return text for Seven Segment Display and English
def process_file_thresh(input_file, size):
    """
    Processes an input image file by resizing it, converting it to grayscale, and applying binary thresholding.
    Extracts text using Tesseract OCR with both Seven Segment and English character training data.

    Args:
        input_file (str): The path to the input image file.
        size (int): The desired size of the output image after resizing.

    Returns:
        A tuple containing the extracted text as strings: (text_ssd, text_eng).
    """

    # Get image data
    # img = read_input_data(input_file)
    img = read_resize_data(input_file, size)

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
def process_file_closing(input_file, size):

    """
    Processes an input image file by resizing it, converting it to grayscale, and applying a closing operation
    using dilation and erosion with 5x5 matrices. Extracts text using Tesseract OCR with both Seven Segment and English
    character training data.

    Args:
        input_file (str): The path to the input image file.
        size (int): The desired size of the output image after resizing.

    Returns:
        A tuple containing the extracted text as strings: (text_ssd, text_eng).
    """

    # Get image data
    # img = read_input_data(input_file)
    img = read_resize_data(input_file, size)

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
def process_adaptive_gaussian(input_file, size):

    """
    Processes an input image file by resizing it, converting it to grayscale, and applying adaptive thresholding
    using a Gaussian-weighted sum of the neighbourhood pixel values. Extracts text using Tesseract OCR with both
    Seven Segment and English character training data.

    Args:
        input_file (str): The path to the input image file.
        size (int): The desired size of the output image after resizing.

    Returns:
        A tuple containing the extracted text as strings: (text_ssd, text_eng).
    """

    # Get image data
    # img = read_input_data(input_file)
    img = read_resize_data(input_file, size)

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
def process_otsu(input_file, size):

    """
    Processes an input image file by resizing it, converting it to grayscale, and applying Otsu's thresholding method
    for image segmentation. Extracts text using Tesseract OCR with both Seven Segment and English character training
    data.

    Args:
        input_file (str): The path to the input image file.
        size (int): The desired size of the output image after resizing.

    Returns:
        A tuple containing the extracted text as strings: (text_ssd, text_eng).
    """

    # Get image data
    img = read_resize_data(input_file, size)

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
