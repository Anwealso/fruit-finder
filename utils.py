"""
utils.py

30/10/2022

Contains extra utility functions to help with things like plotting charts, webcam streaming, recording output examples, etc.

"""

from pickle import FRAME
import cv2
import PIL
from PIL import ImageFont, ImageColor
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
import glob as glob


def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, font=PIL.ImageFont.load_default(), thickness=4, display_str_list=()):
    """
    Adds a single bounding box to an image.

        Parameters:
            image (???): An image
            ymin (int): Top edge limit of the box
            xmin (int): Left edge limit of the box
            ymax (int): Bottom edge limit of the box
            xmax (int): Right edge limit of the box
            color (???): Bounding box color
            (optional) font (???): Label font
            (optional) thickness (???): Bounding box thickness
            (optional) display_str_list (???): ??????

        Returns:
            overlayed_image (ndarray): The image with the prediction results overlayed
    """

    draw = PIL.ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                (left, top)],
            width=thickness,
            fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height

    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                        fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                    display_str,
                    fill="black",
                    font=font)
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """
    Overlay labeled boxes on an image with formatted scores and label names.

        Parameters:
            image (???): An image
            boxes (???): List of bounding boxes in ??? format
            class_names (???): List of bounding boxes in ??? format
            scores (???): Confidence scores of each box
            (optional) max_boxes (???): Max number of boxes to draw
            (optional) min_score (???): Minimum confidence score of a prediction for us to plot its box on the image

        Returns:
            overlayed_image (ndarray): The image with the prediction results overlayed
    """

    colors = list(PIL.ImageColor.colormap.values())

    try:
        font = PIL.ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                                25)
    except IOError:
        print("Font not found, using default font.")
        font = PIL.ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"), int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = PIL.Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image( image_pil, ymin, xmin, ymax, xmax, color, font, display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))

    return image


def plot_predictions(image, predictions):
    """
    Plots a list of bounding boxes and class predictions overlayed on an image

        Parameters:
            frame (ndarray): An image
            predictions (dict): List of ["boundary_box": [(x1,y1), (x2,y2)], "class":"classname"] objects / dicts???

        Returns:
            overlayed_image (ndarray): The image with the prediction results overlayed
    """

    # TODO: Implement ...


def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

def display_image(image):
    fig = plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)

def run_detector(detector, input_folder, output_folder, verbose=True):
    for path in glob.glob(input_folder + '*.*'): # get any file in that folder
        img = load_img(path)

        converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
        start_time = time.time()
        result = detector(converted_img)
        end_time = time.time()

        result = {key:value.numpy() for key,value in result.items()}

        if verbose == True:
            print("Found %d objects." % len(result["detection_scores"]))
            print("Inference time: ", end_time-start_time)

        image_with_boxes = draw_boxes(
            img.numpy(), result["detection_boxes"],
            result["detection_class_entities"], result["detection_scores"])

        # Save the output image
        output_path = output_folder + "out_" + "/".join(path.split("/")[-1:])
        plt.savefig(output_path)

        if verbose == True:
            # Show the output image
            fig = plt.figure(figsize=(20, 15))
            plt.grid(False)
            plt.imshow(image_with_boxes)
            plt.close()


def view_webcam():
    """
    Shows a stream of the output of the webcam

        Parameters:
            None

        Returns:
            None
    """

    print("Connecting to webcam ...")
    # Define a video capture object
    vid = cv2.VideoCapture(0)
    
    print("Streaming frames from webcam ...")
    while(True):
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
    
        # Display the resulting frame
        cv2.imshow('frame', frame)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print("Disconnecting from webcam ...")
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def show_detection_webcam(model, label_map):
    """
    Shows the detection of objects by the model on the live webcam videp

        Parameters:
            trained_model (Keras Model): The trained object detection model
            label_map (dict): A mapping of the detector output values to their corresponding class names

        Returns:
            None
    """

    print("Connecting to webcam ...")
    # Define a video capture object
    vid = cv2.VideoCapture(0)
    
    print("Streaming frames from webcam ...")
    while(True):
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
    
        # Run inference on the frame using the model to get a list of boundary box and class predictions
        predictions = model.predict(frame)
        
        # predictions = list of ["boundary_box": [(x1,y1), (x2,y2)], "class":"classname"] objects / dicts

        out_frame = plot_predictions(frame, predictions)

        # Display the resulting frame
        cv2.imshow('out_frame', out_frame)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print("Disconnecting from webcam ...")
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
