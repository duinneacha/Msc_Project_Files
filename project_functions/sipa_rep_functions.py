from reportlab.platypus import Table, Image
from reportlab.lib import colors
import os
import io
import numpy as np
import sys

from PIL import Image as PILImage

sys.path.append("../../Seperate_Folders/functions/")
import ad_functions as adfns
import sipa08_functions as s8fns

# os.chdir(r"d:/MTU/_Project_Grunt_Work/pdf_test/sipa_08")

###############################################################################################################################################
def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
        print(f"{filename} has been deleted.")
    else:
        print(f"{filename} does not exist.")


###############################################################################################################################################
def genHeaderTable(width: float, height: float, title: str) -> Table:

    # left_image_filename = "mtu_logo.png"
    left_image_filename = "mtu_logo-removebg-preview.png"
    right_image_filename = "nimbus_logo.png"

    left_image_width_ratio = 0.15
    heading_text_ratio = 0.70
    right_image_width_ratio = 0.15

    # Calculate the widths of the columns
    left_img_width = width * left_image_width_ratio
    heading_text_width = width * heading_text_ratio
    right_img_width = width * right_image_width_ratio

    width_list = [left_img_width, heading_text_width, right_img_width]

    # Set the path and width for the left image
    left_img_path = os.path.join("resources", left_image_filename)
    left_img = Image(left_img_path, width=80, height=80, kind="proportional")

    # Set the path and width for the right image
    right_img_path = os.path.join("resources", right_image_filename)
    # right_img = Image(right_img_path, right_img_width, height, kind="proportional")
    right_img = Image(right_img_path, width=80, height=80, kind="proportional")

    # Set the text for the title
    title_text = title

    # Generate the table with the left image, right image, and title text
    table_data = [[left_img, title_text, right_img]]
    table = Table(table_data, width_list, height)

    # title_text_bottom_padding = 40
    color = colors.HexColor("#A9A9A9")

    table.setStyle(
        [
            # ("GRID", (0, 0), (-1, -1), 1, "red"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),  # horizontal
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),  # vertical
            ("BACKGROUND", (0, 0), (-1, -1), color),
            ("FONTSIZE", (1, 0), (1, 0), 20),
        ]
    )

    return table


###############################################################################################################################################
def sipa08_genBodyTable(
    width, height, image_matrix, digits_matrix, file_list, chunkSize
):

    # print(os.getcwd())
    # print(file_list)
    column_headers = [
        "Original",
        "Masked",
        "Deblurred",
        "Thresh",
        "MOtsu",
        "Dst",
        "Dot Matrix",
    ]

    widthList = [
        width * 12 / 100,
        width * 12 / 100,
        width * 12 / 100,
        width * 12 / 100,
        width * 12 / 100,
        width * 12 / 100,
        width * 12 / 100,
    ]
    # print("digits_matrix: ", digits_matrix)
    rowCount = 6

    matrix = [
        [None for col in range(image_matrix.shape[1] + 1)]
        for row in range((image_matrix.shape[0] * 2) + 1)
    ]

    # Update the first row with the column headers
    for i, header in enumerate(column_headers):
        matrix[0][i + 1] = header

    for row in range(0, image_matrix.shape[0]):
        for col in range(0, image_matrix.shape[1]):
            # print("row: ", row, "col: ", col)
            if row == 0:
                matrix[row + 1][0] = file_list[0]
                matrix[row + 2][0] = f"Lets\nSSD\nEng\nDot Matrix"
            else:
                matrix[2 + 1][0] = file_list[1]
                matrix[3 + 1][0] = f"Lets\nSSD\nEng\nDot Matrix"
            # print("file_list: ", file_list[0])
            img_data = s8fns.format_image_for_pdf(image_matrix[row, col])

            # Create ReportLab image from the bytes
            img = Image(img_data, width=60, height=60, kind="proportional")
            matrix[(row * 2) + 1][col + 1] = img

            # Add the content of digits_matrix below the corresponding image
            digits_list_1 = digits_matrix[row, col * 4]
            digits_list_2 = digits_matrix[row, col * 4 + 1]
            digits_list_3 = digits_matrix[row, col * 4 + 2]
            digits_list_4 = digits_matrix[row, col * 4 + 3]

            digits_str_1 = "".join(digits_list_1)
            digits_str_2 = "".join(digits_list_2)
            digits_str_3 = "".join(digits_list_3)
            digits_str_4 = "".join(digits_list_4)

            final_digits_str = (
                f"{digits_str_1}\n{digits_str_2}\n{digits_str_3}\n{digits_str_4}"
            )
            matrix[row * 2 + 2][col + 1] = final_digits_str

    # matrix[0][0] == "ADADAD"
    # print("matrix", matrix)
    # print("matrix[0][0]", matrix[0][0])
    # print("matrix[0][1]", matrix[0][1])
    body_table = Table(matrix, widthList, height / rowCount)

    color = colors.toColor("rgba(0, 115, 153, 0.9)")
    header_colour = colors.toColor("rgb(186, 232, 207)")
    body_table.setStyle(
        [
            ("INNERGRID", (0, 0), (-1, -1), 0.5, "black"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), ["antiquewhite", "beige"]),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            # ("RIGHTPADDING", (0, 0), (-1, -1), 20),
            # ("GRID", (0, 0), (-1, -1), 1, "red"),
            # ("FONTSIZE", (0, 0), (-1, 0), 12),
            # ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            # ("ALIGN", (1, 1), (2, -1), "CENTER"),
            # ("ALIGN", (5, 1), (5, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            # ("BACKGROUND", (0, 0), (-1, 0), color),
            # ("TEXTCOLOR", (0, 0), (-1, 0), "white"),
            ("BACKGROUND", (1, 0), (-1, 0), header_colour),
            ("FONTNAME", (1, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 1), (0, -1), header_colour),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ]
    )

    return body_table


###############################################################################################################################################


###############################################################################################################################################
def genFooterTable(width, height):
    text = "R00145278 Aidan Dennehy, MSc. Project"

    footer_section = Table([[text]], width, height)

    color = colors.HexColor("#A9A9A9")

    footer_section.setStyle(
        [
            # ('GRID', (0,0), (-1,-1), 1, 'red'),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("BACKGROUND", (0, 0), (-1, -1), color),
            ("TEXTCOLOR", (0, 0), (-1, -1), "black"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),  # horizontal align
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),  # vertical align
        ]
    )

    return footer_section


def genADADBodyTable(width, height, image_matrix, chunkSize):

    widthList = [width * 12 / 100, width * 12 / 100, width * 12 / 100]
    matrix = [
        ["", "", ""],
        ["", "", ""],
        ["", "", ""],
        ["", "", ""],
        ["", "", ""],
    ]
    rows = 5
    cols = 3
    rowCount = 2

    # Create an empty list to store the images
    # matrix = [
    #     [None for col in range(image_matrix.shape[1])]
    #     for row in range(image_matrix.shape[0])
    # ]

    print("image_matrix.shape: ", image_matrix.shape)

    for row in range(image_matrix.shape[0]):
        print("row: ", row)
        for col in range(image_matrix.shape[1]):
            print(" -- col: ", col)

            img_data = image_matrix[row, col].astype(np.uint8)

            # if adfns.is_image(img_data):
            #     print(" -- -- is_image: ", img_data)

            # Convert NumPy array to PIL image
            # pil_image = PILImage.fromarray(img_data)

            # print("pil_image: ", pil_image.size)

            # Convert PIL image to bytes
            # with io.BytesIO() as output:
            # pil_image.save(output, format="PNG")
            # img_bytes = output.getvalue()

            # Create ReportLab image from the bytes
            # img = Image(img_bytes, width=60, height=60, kind="proportional")
            # img = Image(img_data, width=60, height=60, kind="proportional")
            img = PILImage.fromarray(img_data)
            # adfns.show_img(img_data, 3)

            # rp_image = Image(img_data, width=60, height=60, kind="proportional")

            # matrix[row][col] = img_data

    img_formatted = s8fns.format_image_for_pdf(image_matrix[0][0])

    # temp_img = image_matrix[0][0].astype(np.uint8)
    # img_data = PILImage.fromarray(temp_img)
    # img_data_png = io.BytesIO()
    # img_data.save(img_data_png, format="PNG")
    # img_data_png.seek(0)

    # ad_image = Image()
    # ad_image.setImage(img_data, width=60, height=60, kind="proportional")

    img1 = Image(img_formatted, width=160, height=160, kind="proportional")
    # img2 = Image(image_matrix[0][1], width=60, height=60, kind="proportional")
    # img3 = Image(image_matrix[1][0], width=60, height=60, kind="proportional")
    # img4 = Image(image_matrix[1][1], width=60, height=60, kind="proportional")

    matrix[1][0] = img1
    # matrix[0][1] = img2
    # matrix[1][0] = img3
    # matrix[1][1] = img4

    body_table = Table(matrix, widthList, height / 3)

    color = colors.toColor("rgba(0, 115, 153, 0.9)")

    body_table.setStyle(
        [
            ("INNERGRID", (0, 0), (-1, -1), 0.5, "grey"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), ["antiquewhite", "beige"]),
            # ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            # ("LEFTPADDING", (0, 0), (-1, -1), 60),
            # ("RIGHTPADDING", (0, 0), (-1, -1), 20),
            # ("GRID", (0, 0), (-1, -1), 1, "red"),
            # ("FONTSIZE", (0, 0), (-1, 0), 12),
            # ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            # ("ALIGN", (1, 1), (2, -1), "CENTER"),
            # ("ALIGN", (5, 1), (5, -1), "RIGHT"),
            # ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            # ("BACKGROUND", (0, 0), (-1, 0), color),
            # ("TEXTCOLOR", (0, 0), (-1, 0), "white"),
        ]
    )
    # print("res: ", res)
    return body_table


def test_body(width, height, image_matrix, chunkSize):

    return "test_body"


############################################################################################################
def s09_genBodyTable(width, height, image_matrix, digits_matrix, file_list, chunkSize):

    # print(os.getcwd())
    # print(file_list)
    column_headers = [
        "Original",
        "Deblurred",
        "Thresh",
        "Dot Matrix",
        "Inverted Thresh",
    ]

    widthList = [
        width * 12 / 100,
        width * 12 / 100,
        width * 12 / 100,
        width * 12 / 100,
        width * 12 / 100,
        width * 12 / 100,
        width * 12 / 100,
    ]
    # print("digits_matrix: ", digits_matrix)
    rowCount = 6

    matrix = [
        [None for col in range(image_matrix.shape[1] + 1)]
        for row in range((image_matrix.shape[0] * 2) + 1)
    ]
    print("matrix: ", matrix)
    # Update the first row with the column headers
    for i, header in enumerate(column_headers):
        matrix[0][i + 1] = header

    for row in range(0, image_matrix.shape[0]):
        for col in range(0, image_matrix.shape[1]):
            # print("row: ", row, "col: ", col)
            if row == 0:
                matrix[row + 1][0] = file_list[0]
                matrix[row + 2][0] = f"Lets\nSSD\nEng\nDot Matrix"
            else:
                matrix[2 + 1][0] = file_list[1]
                matrix[3 + 1][0] = f"Lets\nSSD\nEng\nDot Matrix"
            # print("file_list: ", file_list[0])
            img_data = s8fns.format_image_for_pdf(image_matrix[row, col])

            # Create ReportLab image from the bytes
            img = Image(img_data, width=60, height=60, kind="proportional")
            matrix[(row * 2) + 1][col + 1] = img

            # Add the content of digits_matrix below the corresponding image
            digits_list_1 = digits_matrix[row, col * 4]
            digits_list_2 = digits_matrix[row, col * 4 + 1]
            digits_list_3 = digits_matrix[row, col * 4 + 2]
            digits_list_4 = digits_matrix[row, col * 4 + 3]

            digits_str_1 = "".join(digits_list_1)
            digits_str_2 = "".join(digits_list_2)
            digits_str_3 = "".join(digits_list_3)
            digits_str_4 = "".join(digits_list_4)

            final_digits_str = (
                f"{digits_str_1}\n{digits_str_2}\n{digits_str_3}\n{digits_str_4}"
            )
            matrix[row * 2 + 2][col + 1] = final_digits_str

    # matrix[0][0] == "ADADAD"
    # print("matrix", matrix)
    # print("matrix[0][0]", matrix[0][0])
    # print("matrix[0][1]", matrix[0][1])
    body_table = Table(matrix, widthList, height / rowCount)

    color = colors.toColor("rgba(0, 115, 153, 0.9)")
    header_colour = colors.toColor("rgb(186, 232, 207)")
    body_table.setStyle(
        [
            ("INNERGRID", (0, 0), (-1, -1), 0.5, "black"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), ["antiquewhite", "beige"]),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            # ("RIGHTPADDING", (0, 0), (-1, -1), 20),
            # ("GRID", (0, 0), (-1, -1), 1, "red"),
            # ("FONTSIZE", (0, 0), (-1, 0), 12),
            # ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            # ("ALIGN", (1, 1), (2, -1), "CENTER"),
            # ("ALIGN", (5, 1), (5, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            # ("BACKGROUND", (0, 0), (-1, 0), color),
            # ("TEXTCOLOR", (0, 0), (-1, 0), "white"),
            ("BACKGROUND", (1, 0), (-1, 0), header_colour),
            ("FONTNAME", (1, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 1), (0, -1), header_colour),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ]
    )

    return body_table


############################################################################################################
def s09_crnn_genBodyTable(
    width, height, image_matrix, digits_matrix, file_list, chunkSize
):

    # print(os.getcwd())
    # print(file_list)
    column_headers = [
        "Original",
        "Inverted Thresh",
        "CRNN 1",
        "CRNN 2",
        "CRNN 3",
        "CRNN 4",
        "CRNN 5",
        "CRNN 6",
    ]

    widthList = [
        width * 10 / 100,
        width * 10 / 100,
        width * 10 / 100,
        width * 10 / 100,
        width * 10 / 100,
        width * 10 / 100,
        width * 10 / 100,
        width * 10 / 100,
    ]
    # print("digits_matrix: ", digits_matrix)
    rowCount = 6

    matrix = [
        [None for col in range(image_matrix.shape[1] + 1)]
        for row in range((image_matrix.shape[0] * 2) + 1)
    ]
    print("matrix: ", matrix)
    # Update the first row with the column headers
    for i, header in enumerate(column_headers):
        matrix[0][i + 1] = header

    for row in range(0, image_matrix.shape[0]):
        for col in range(0, image_matrix.shape[1]):
            # print("row: ", row, "col: ", col)
            if row == 0:
                matrix[row + 1][0] = file_list[0]
                matrix[row + 2][0] = f"Lets\nSSD\nEng\nDot Matrix"
            else:
                matrix[2 + 1][0] = file_list[1]
                matrix[3 + 1][0] = f"Lets\nSSD\nEng\nDot Matrix"
            # print("file_list: ", file_list[0])
            img_data = s8fns.format_image_for_pdf(image_matrix[row, col])

            # Create ReportLab image from the bytes
            img = Image(img_data, width=60, height=60, kind="proportional")
            matrix[(row * 2) + 1][col + 1] = img

            # Add the content of digits_matrix below the corresponding image
            digits_list_1 = str(digits_matrix[row, col * 4])
            digits_list_2 = str(digits_matrix[row, col * 4 + 1])
            digits_list_3 = str(digits_matrix[row, col * 4 + 2])
            digits_list_4 = str(digits_matrix[row, col * 4 + 3])

            print("type(digits_list_1)", type(digits_list_1))

            # digits_str_1 = "".join(str(digit) for digit in digits_list_1)
            # digits_str_2 = "".join(str(digit) for digit in digits_list_3)
            # digits_str_3 = "".join(str(digit) for digit in digits_list_3)
            # digits_str_4 = "".join(str(digit) for digit in digits_list_4)

            digits_str_1 = "".join(digits_list_1)
            digits_str_2 = "".join(digits_list_2)
            digits_str_3 = "".join(digits_list_3)
            digits_str_4 = "".join(digits_list_4)

            final_digits_str = (
                f"{digits_str_1}\n{digits_str_2}\n{digits_str_3}\n{digits_str_4}"
            )
            matrix[row * 2 + 2][col + 1] = final_digits_str

    # matrix[0][0] == "ADADAD"
    # print("matrix", matrix)
    # print("matrix[0][0]", matrix[0][0])
    # print("matrix[0][1]", matrix[0][1])
    body_table = Table(matrix, widthList, height / rowCount)

    color = colors.toColor("rgba(0, 115, 153, 0.9)")
    header_colour = colors.toColor("rgb(186, 232, 207)")
    body_table.setStyle(
        [
            ("INNERGRID", (0, 0), (-1, -1), 0.5, "black"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), ["antiquewhite", "beige"]),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            # ("RIGHTPADDING", (0, 0), (-1, -1), 20),
            # ("GRID", (0, 0), (-1, -1), 1, "red"),
            # ("FONTSIZE", (0, 0), (-1, 0), 12),
            # ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            # ("ALIGN", (1, 1), (2, -1), "CENTER"),
            # ("ALIGN", (5, 1), (5, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            # ("BACKGROUND", (0, 0), (-1, 0), color),
            # ("TEXTCOLOR", (0, 0), (-1, 0), "white"),
            ("BACKGROUND", (1, 0), (-1, 0), header_colour),
            ("FONTNAME", (1, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 1), (0, -1), header_colour),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ]
    )

    return body_table
