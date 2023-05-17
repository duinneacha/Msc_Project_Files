from reportlab.graphics.charts.barcharts import VerticalBarChart

# from reportlab.graphics.renderPDF import draw
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Drawing
import io

###############################################################################################################################################
def create_pdf_document(filename, pagesize):
    """
    Creates a new PDF document with the specified filename and pagesize.

    :param filename: The name of the PDF file to create.
    :param pagesize: The size of the PDF pages, specified as a tuple of (width, height).
    :return: A ReportLab canvas object representing the new PDF document.
    """
    canvas_obj = canvas.Canvas(filename, pagesize=pagesize)
    return canvas_obj


###############################################################################################################################################
def add_header_footer(canvas_obj, header_text, footer_text):
    """
    Adds a header and footer to each page of the PDF document.

    :param canvas_obj: The ReportLab canvas object representing the PDF document.
    :param header_text: The text to include in the header.
    :param footer_text: The text to include in the footer.
    """
    canvas_obj.setTitle(header_text)
    canvas_obj.drawCentredString(
        canvas_obj._pagesize[0] / 2.0, canvas_obj._pagesize[1] - 20, header_text
    )
    canvas_obj.drawCentredString(canvas_obj._pagesize[0] / 2.0, 20, footer_text)


def add_text(canvas_obj, text, x, y, font_size=12):
    """
    Adds text to the PDF document at the specified position.

    :param canvas_obj: The ReportLab canvas object representing the PDF document.
    :param text: The text to add to the document.
    :param x: The x-coordinate of the position where the text should be added, specified in points.
    :param y: The y-coordinate of the position where the text should be added, specified in points.
    :param font_size: The font size to use for the text.
    """
    canvas_obj.setFont("Helvetica", font_size)
    canvas_obj.drawString(x, y, text)


###############################################################################################################################################
def add_image(canvas_obj, image_path, x, y, width=None, height=None):
    """
    Adds an image to the PDF document at the specified position.

    :param canvas_obj: The ReportLab canvas object representing the PDF document.
    :param image_path: The path to the image file to add to the document.
    :param x: The x-coordinate of the position where the image should be added, specified in points.
    :param y: The y-coordinate of the position where the image should be added, specified in points.
    :param width: The width of the image to be added, specified in points. If not provided, the image's original width will be used.
    :param height: The height of the image to be added, specified in points. If not provided, the image's original height will be used.
    """
    print("image_path", image_path)
    canvas_obj.drawImage(image_path, x, y, width=width, height=height)


###############################################################################################################################################
# def add_bar_chart(canvas_obj, data, x, y, width, height):
#     """
#     Adds a bar chart to the PDF document at the specified position.

#     :param canvas_obj: The ReportLab canvas object representing the PDF document.
#     :param data: A list of data points to be graphed.
#     :param x: The x-coordinate of the position where the graph should be added, specified in points.
#     :param y: The y-coordinate of the position where the graph should be added, specified in points.
#     :param width: The width of the graph, specified in points.
#     :param height: The height of the graph, specified in points.
#     """
#     chart = VerticalBarChart()
#     chart.width = width
#     chart.height = height
#     chart.data = [data]
#     chart.barLabelFormat = "%d"
#     chart.valueAxis.labelTextFormat = "%d"
#     chart.valueAxis.valueMin = 0
#     chart.categoryAxis.labels.boxAnchor = "ne"
#     chart.categoryAxis.labels.angle = 45
#     chart.categoryAxis.labels.dy = -10
#     chart.categoryAxis.categoryNames = ["Data"]
#     chart.bars[0].fillColor = colors.blue
#     chart.bars[0].strokeColor = colors.black
#     chart.bars[0].strokeWidth = 1
#     chart.drawOn(canvas_obj, x, y)


# def add_bar_chart(canvas_obj, data, x, y, width, height):
#     """
#     Adds a bar chart to the PDF document at the specified position.

#     :param canvas_obj: The ReportLab canvas object representing the PDF document.
#     :param data: A list of data points to be graphed.
#     :param x: The x-coordinate of the position where the graph should be added, specified in points.
#     :param y: The y-coordinate of the position where the graph should be added, specified in points.
#     :param width: The width of the graph, specified in points.
#     :param height: The height of the graph, specified in points.
#     """
#     chart = VerticalBarChart()
#     chart.width = width
#     chart.height = height
#     chart.data = [data]
#     chart.barLabelFormat = "%d"
#     chart.valueAxis.labelTextFormat = "%d"
#     chart.valueAxis.valueMin = 0
#     chart.categoryAxis.labels.boxAnchor = "ne"
#     chart.categoryAxis.labels.angle = 45
#     chart.categoryAxis.labels.dy = -10
#     chart.categoryAxis.categoryNames = ["Data"]
#     chart.bars[0].fillColor = colors.blue
#     chart.bars[0].strokeColor = colors.black
#     chart.bars[0].strokeWidth = 1
#     chart.draw()
#     chart_x, chart_y = chart.getPosition()
#     draw(chart, canvas_obj, x, y - chart_y)


def add_bar_chart(canvas_obj, data, x, y, width, height):
    """
    Adds a bar chart to the PDF document at the specified position.

    :param canvas_obj: The ReportLab canvas object representing the PDF document.
    :param data: A list of data points to be graphed.
    :param x: The x-coordinate of the position where the graph should be added, specified in points.
    :param y: The y-coordinate of the position where the graph should be added, specified in points.
    :param width: The width of the graph, specified in points.
    :param height: The height of the graph, specified in points.
    """
    chart = VerticalBarChart()
    chart.width = width
    chart.height = height
    chart.data = [data]
    chart.barLabelFormat = "%d"
    chart.valueAxis.labelTextFormat = "%d"
    chart.valueAxis.valueMin = 0
    chart.categoryAxis.labels.boxAnchor = "ne"
    chart.categoryAxis.labels.angle = 45
    chart.categoryAxis.labels.dy = -10
    chart.categoryAxis.categoryNames = ["Data"]
    chart.bars[0].fillColor = colors.blue
    chart.bars[0].strokeColor = colors.black
    chart.bars[0].strokeWidth = 1

    # Create a drawing object to add the chart to
    drawing = Drawing(width, height)
    drawing.add(chart)

    # Render the drawing and add it to the PDF document
    renderPDF.draw(drawing, canvas_obj, x, y - height)


###############################################################################################################################################
def close_pdf_document(canvas_obj):
    """
    Closes the PDF document and saves the contents to the file.

    :param canvas_obj: The ReportLab canvas object representing the PDF document.
    """
    canvas_obj.save()


###############################################################################################################################################
def add_layered_canvas(canvas_obj):
    """
    Adds a layered canvas to the PDF document.

    :param canvas_obj: The ReportLab canvas object representing the PDF document.
    :return: A list of two canvas objects representing the bottom and top layers of the layered canvas.
    """
    bottom_canvas = canvas_obj
    top_canvas = canvas.Canvas(canvas_obj._pagesize)
    return [bottom_canvas, top_canvas]


# def close_layered_canvas(layered_canvas):
#     """
#     Closes a layered canvas and adds it to the PDF document in the correct order.

#     :param layered_canvas: A list of two canvas objects representing the bottom and top layers of the layered canvas.
#     """
#     bottom_canvas, top_canvas = layered_canvas

#     # Move the cursor to the top layer of the canvas
#     top_canvas.showPage()

#     # Copy the top layer to the bottom layer
#     top_canvas.save()
#     bottom_canvas.drawInlineImage(
#         top_canvas, 0, 0, bottom_canvas._pagesize[0], bottom_canvas._pagesize[1]
#     )
#     bottom_canvas.showPage()

#     # Close the canvas objects
#     top_canvas.save()
#     bottom_canvas.save()

###############################################################################################################################################
# def close_layered_canvas(layered_canvas):
#     bottom_canvas, top_canvas = layered_canvas

#     # Save the top layer to a StringIO buffer
#     top_canvas_buffer = io.BytesIO()
#     top_canvas.save(top_canvas_buffer)

#     # Create a new canvas object for the bottom layer
#     bottom_canvas.showPage()
#     bottom_canvas_buffer = io.BytesIO()
#     bottom_canvas.save(bottom_canvas_buffer)

#     # Copy the top layer to the bottom layer
#     top_canvas_buffer.seek(0)
#     bottom_canvas.drawInlineImage(
#         top_canvas_buffer, 0, 0, bottom_canvas._pagesize[0], bottom_canvas._pagesize[1]
#     )
#     bottom_canvas.showPage()

#     # Close the canvas objects
#     bottom_canvas.save()
#     top_canvas.save()

#     # Get the PDF data from the bottom layer
#     bottom_canvas_buffer.seek(0)
#     pdf_data = bottom_canvas_buffer.getvalue()

#     # Write the PDF data to the file
#     canvas_obj = bottom_canvas.getpdfdata()
#     with open(canvas_obj.filename, "wb") as f:
#         f.write(pdf_data)


# def close_layered_canvas(layered_canvas):
#     bottom_canvas, top_canvas = layered_canvas

#     # Save the top layer to a StringIO buffer
#     top_canvas_buffer = io.BytesIO()
#     top_canvas.save(top_canvas_buffer)

#     # Create a new canvas object for the bottom layer
#     bottom_canvas.showPage()
#     bottom_canvas_buffer = io.BytesIO()
#     bottom_canvas.save(bottom_canvas_buffer)

#     # Copy the top layer to the bottom layer
#     top_canvas_buffer.seek(0)
#     bottom_canvas.drawInlineImage(
#         top_canvas_buffer, 0, 0, bottom_canvas._pagesize[0], bottom_canvas._pagesize[1]
#     )
#     bottom_canvas.showPage()

#     # Close the canvas objects
#     bottom_canvas.save()
#     top_canvas.save()

#     # Get the PDF data from the bottom layer
#     bottom_canvas_buffer.seek(0)
#     pdf_data = bottom_canvas_buffer.getvalue()

#     # Write the PDF data to the file
#     with open(bottom_canvas.filename, "wb") as f:
#         f.write(pdf_data)


def close_layered_canvas(layered_canvas):
    bottom_canvas, top_canvas = layered_canvas

    # Save the top layer to a StringIO buffer
    top_canvas_buffer = io.BytesIO()
    top_canvas.save()
    top_canvas_buffer.write(top_canvas.getPage(1).pdf)

    # Create a new canvas object for the bottom layer
    bottom_canvas.showPage()
    bottom_canvas_buffer = io.BytesIO()
    bottom_canvas.save(bottom_canvas_buffer)

    # Copy the top layer to the bottom layer
    top_canvas_buffer.seek(0)
    bottom_canvas.drawInlineImage(
        top_canvas_buffer, 0, 0, bottom_canvas._pagesize[0], bottom_canvas._pagesize[1]
    )
    bottom_canvas.showPage()

    # Close the canvas objects
    bottom_canvas.save()
    top_canvas.save()

    # Get the PDF data from the bottom layer
    bottom_canvas_buffer.seek(0)
    pdf_data = bottom_canvas_buffer.getvalue()

    # Write the PDF data to the file
    with open(bottom_canvas.filename, "wb") as f:
        f.write(pdf_data)
