"""
utils.py

30/10/2022

Contains extra utility functions to help with things like plotting charts, webcam streaming, recording output examples, etc.

"""

from pickle import FRAME
import cv2
import PIL
from PIL import ImageFont, ImageColor, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
import glob as glob


# Load the COCO Label Map
category_index = {
    1: {'id': 1, 'name': 'person'},
    2: {'id': 2, 'name': 'bicycle'},
    3: {'id': 3, 'name': 'car'},
    4: {'id': 4, 'name': 'motorcycle'},
    5: {'id': 5, 'name': 'airplane'},
    6: {'id': 6, 'name': 'bus'},
    7: {'id': 7, 'name': 'train'},
    8: {'id': 8, 'name': 'truck'},
    9: {'id': 9, 'name': 'boat'},
    10: {'id': 10, 'name': 'traffic light'},
    11: {'id': 11, 'name': 'fire hydrant'},
    13: {'id': 13, 'name': 'stop sign'},
    14: {'id': 14, 'name': 'parking meter'},
    15: {'id': 15, 'name': 'bench'},
    16: {'id': 16, 'name': 'bird'},
    17: {'id': 17, 'name': 'cat'},
    18: {'id': 18, 'name': 'dog'},
    19: {'id': 19, 'name': 'horse'},
    20: {'id': 20, 'name': 'sheep'},
    21: {'id': 21, 'name': 'cow'},
    22: {'id': 22, 'name': 'elephant'},
    23: {'id': 23, 'name': 'bear'},
    24: {'id': 24, 'name': 'zebra'},
    25: {'id': 25, 'name': 'giraffe'},
    27: {'id': 27, 'name': 'backpack'},
    28: {'id': 28, 'name': 'umbrella'},
    31: {'id': 31, 'name': 'handbag'},
    32: {'id': 32, 'name': 'tie'},
    33: {'id': 33, 'name': 'suitcase'},
    34: {'id': 34, 'name': 'frisbee'},
    35: {'id': 35, 'name': 'skis'},
    36: {'id': 36, 'name': 'snowboard'},
    37: {'id': 37, 'name': 'sports ball'},
    38: {'id': 38, 'name': 'kite'},
    39: {'id': 39, 'name': 'baseball bat'},
    40: {'id': 40, 'name': 'baseball glove'},
    41: {'id': 41, 'name': 'skateboard'},
    42: {'id': 42, 'name': 'surfboard'},
    43: {'id': 43, 'name': 'tennis racket'},
    44: {'id': 44, 'name': 'bottle'},
    46: {'id': 46, 'name': 'wine glass'},
    47: {'id': 47, 'name': 'cup'},
    48: {'id': 48, 'name': 'fork'},
    49: {'id': 49, 'name': 'knife'},
    50: {'id': 50, 'name': 'spoon'},
    51: {'id': 51, 'name': 'bowl'},
    52: {'id': 52, 'name': 'banana'},
    53: {'id': 53, 'name': 'apple'},
    54: {'id': 54, 'name': 'sandwich'},
    55: {'id': 55, 'name': 'orange'},
    56: {'id': 56, 'name': 'broccoli'},
    57: {'id': 57, 'name': 'carrot'},
    58: {'id': 58, 'name': 'hot dog'},
    59: {'id': 59, 'name': 'pizza'},
    60: {'id': 60, 'name': 'donut'},
    61: {'id': 61, 'name': 'cake'},
    62: {'id': 62, 'name': 'chair'},
    63: {'id': 63, 'name': 'couch'},
    64: {'id': 64, 'name': 'potted plant'},
    65: {'id': 65, 'name': 'bed'},
    67: {'id': 67, 'name': 'dining table'},
    70: {'id': 70, 'name': 'toilet'},
    72: {'id': 72, 'name': 'tv'},
    73: {'id': 73, 'name': 'laptop'},
    74: {'id': 74, 'name': 'mouse'},
    75: {'id': 75, 'name': 'remote'},
    76: {'id': 76, 'name': 'keyboard'},
    77: {'id': 77, 'name': 'cell phone'},
    78: {'id': 78, 'name': 'microwave'},
    79: {'id': 79, 'name': 'oven'},
    80: {'id': 80, 'name': 'toaster'},
    81: {'id': 81, 'name': 'sink'},
    82: {'id': 82, 'name': 'refrigerator'},
    84: {'id': 84, 'name': 'book'},
    85: {'id': 85, 'name': 'clock'},
    86: {'id': 86, 'name': 'vase'},
    87: {'id': 87, 'name': 'scissors'},
    88: {'id': 88, 'name': 'teddy bear'},
    89: {'id': 89, 'name': 'hair drier'},
    90: {'id': 90, 'name': 'toothbrush'},
}


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
    font = PIL.ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(category_index[class_names[i]]["name"], int(100 * scores[i]))
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


def run_detector(model, input_folder, output_folder, verbose=True):
    i = 0

    for path in glob.glob(input_folder + '*.*'): # get any file in that folder
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = img.numpy()
        img = np.expand_dims(img, 0)

        start_time = time.time()
        result = model(img)
        end_time = time.time()

        result = {key:value.numpy() for key,value in result.items()}

        image_with_boxes = draw_boxes(
            np.squeeze(img), 
            result["detection_boxes"][0],
            result["detection_classes"][0],
            result["detection_scores"][0],
            max_boxes=10,
            min_score=0.5)

        if verbose == True:
            # Show the output image
            plt.figure(figsize=(20, 15))
            plt.grid(False)
            plt.imshow(image_with_boxes)

        # Save the output image
        output_path = output_folder + "out_" + "/".join(path.split("/")[-1:])
        plt.savefig(output_path)
        plt.close()

        print(i)
        i = i + 1


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
