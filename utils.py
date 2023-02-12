"""
utils.py

30/10/2022

Contains extra utility functions to help with things like plotting charts, webcam streaming, recording output examples, etc.

TODO: Fix documentation

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
import dataset as dataset
import platform as platform
# from playsound import playsound
# import winsound as winsound
from PIL import Image


def plot_detections(image_np, boxes, classes, scores, figsize=(12, 16), image_name=None, min_score=0.5):
    """Wrapper function to visualize detections.

    Args:
        image_np: uint8 numpy array with shape (img_height, img_width, 3)
        boxes: a numpy array of shape [N, 4]
        classes: a numpy array of shape [N]. Note that class indices are 1-based,
        and match the keys in the label map.
        scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
        category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
        figsize: size for the figure.
        image_name: a name for the image file.
    """
    image_np_with_annotations = image_np.copy()

    draw_boxes(
        image_np_with_annotations,
        boxes,
        classes,
        scores,
        min_score=min_score)

    if image_name:
        plt.imsave(image_name, image_np_with_annotations)
        plt.imshow(image_np_with_annotations)
    else:
        plt.imshow(image_np_with_annotations)


def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, thickness=4, display_str_list=()):
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

    # Set the font size to be reasonably readable
    fontsize = int(im_height/20)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', fontsize)
    except IOError:
        font = ImageFont.load_default()

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
    TODO: Fix tiny unreadable text for class labels on large images

        Parameters:
            image (???): An image
            boxes (???): List of bounding boxes in ??? format
            class_names (???): List of class names
            scores (???): Confidence scores of each box
            (optional) max_boxes (???): Max number of boxes to draw
            (optional) min_score (???): Minimum confidence score of a prediction for us to plot its box on the image

        Returns:
            overlayed_image (ndarray): The image with the prediction results overlayed
    """

    # print(f"image (type={type(image)}),  (shape={np.shape(boxes)}): NOT SHOWN FOR CONCISENESS")
    
    colors = list(PIL.ImageColor.colormap.values())

    for i in range(min(len(boxes), max_boxes)):
        # print(f"(box={boxes[i][0]}), score={scores[i]}), class_name={class_names[i]})")

        if scores[i] >= min_score:
            # xmin, ymin, xmax, ymax = tuple(boxes[i][0])
            ymin, xmin, ymax, xmax = tuple(boxes[i][0])
            display_str = "{}: {}%".format(class_names[i], int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = PIL.Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image( image_pil, ymin, xmin, ymax, xmax, color, display_str_list=[display_str])
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


def run_detector_api(model, labels_file, img, min_score=0.5):
    """
    TODO: Document me!
    """

    # Convert the image into the correct format (input 'img' is of PIL Image type)
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, 0)

    # Run inference
    result = model(img)
    result = {key:value.numpy() for key,value in result.items()}

    # Convert Class Indices to Class Names
    labels_dict = dataset.get_tf_labels_from_file(labels_file)
    class_names = [list(labels_dict.keys())[list(labels_dict.values()).index(int(item))] for item in result["detection_classes"][0]]

    image_with_boxes = draw_boxes(
        np.squeeze(img), 
        result["detection_boxes"],
        class_names,
        result["detection_scores"][0],
        max_boxes=10,
        min_score=min_score)

    # Return the output image (converted back to PIL Image type)
    return Image.fromarray(np.uint8(image_with_boxes))


def run_detector(model, input_folder, output_folder, labels_file, min_score=0.5, verbose=True):
    """
    TODO: Document me!
    TODO: Fix the plotting for this function cos i think its real messed up (try copying the implementation of plot_detections)
    
    """

    # TODO: Implement ability to plot multiple boxes on an image

    i = 0
    for path in glob.glob(input_folder + '*.*'): # get any file in that folder
        # Reads in image
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = img.numpy()
        img = np.expand_dims(img, 0)
        
        # Run inference
        result = model(img)
        result = {key:value.numpy() for key,value in result.items()}

        # Convert Class Indices to Class Names
        labels_dict = dataset.get_tf_labels_from_file(labels_file)
        class_names = [list(labels_dict.keys())[list(labels_dict.values()).index(int(item))] for item in result["detection_classes"][0]]

        image_with_boxes = draw_boxes(
            np.squeeze(img), 
            result["detection_boxes"],
            class_names,
            result["detection_scores"][0],
            max_boxes=10,
            min_score=min_score)

        if verbose == True:
            # Show the output image
            plt.figure(figsize=(20, 15))
            plt.grid(False)
            plt.imshow(image_with_boxes)

        # Save the output image
        if platform.system() == 'Windows':
            output_path = output_folder + "out_" + "\\".join(path.split("\\")[-1:])
        else:
            output_path = output_folder + "out_" + "/".join(path.split("/")[-1:])
        plt.savefig(output_path)
        plt.close()

        print(i)
        i = i + 1


def run_detector_live(model, labels_file):
    """
    TODO: Document me!
    
    """

    print("Connecting to webcam ...")
    # Define a video capture object
    vid = cv2.VideoCapture(0)
    
    print("Streaming frames from webcam ...")
    while(True):
        # Capture the video frame by frame
        ret, frame = vid.read()
        img = np.expand_dims(frame, 0)
    
        # Run inference on image
        result = model(img)
        result = {key:value.numpy() for key,value in result.items()}

        # Convert Class Indices to Class Names
        labels_dict = dataset.get_tf_labels_from_file(labels_file)
        class_names = [list(labels_dict.keys())[list(labels_dict.values()).index(int(item))] for item in result["detection_classes"][0]]

        image_with_boxes = draw_boxes(
            np.squeeze(img), 
            result["detection_boxes"][0],
            class_names,
            result["detection_scores"][0],
            max_boxes=10,
            min_score=0.5)

        # Display the resulting frame
        cv2.imshow('image_with_boxes', image_with_boxes)
        
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


def play_beep():
    """
    Plays a beep sound effect (for when the fruit is successfully scanned)
    """

    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 100  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

    # playsound('C:\\Users\\alex\Documents\\dev\\fruit-finder\\bruh.mp3')


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: a file path.

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """

    # Open the image
    img = PIL.Image.open(path)
    # Convert image to numpy array
    data = np.asarray(img)
    # Close the image (not strictly necessary)
    del img

    return data


def get_model_train_step_function(model, optimizer, vars_to_fine_tune):
    """Get a tf.function for training step."""

    # Use tf.function for a bit of speed.
    # Comment out the tf.function decorator if you want the inside of the
    # function to run eagerly.
    @tf.function
    def train_step_fn(image_tensors,
                        groundtruth_boxes_list,
                        groundtruth_classes_list, 
                        batch_size):
        """A single training iteration.

        Args:
        image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
            Note that the height and width can vary across images, as they are
            reshaped within this function to be 640x640.
        groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
            tf.float32 representing groundtruth boxes for each image in the batch.
        groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
            with type tf.float32 representing groundtruth boxes for each image in
            the batch.

        Returns:
        A scalar tensor representing the total loss for the input batch.
        """
        shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)
        model.provide_groundtruth(
            groundtruth_boxes_list=groundtruth_boxes_list,
            groundtruth_classes_list=groundtruth_classes_list)
        with tf.GradientTape() as tape:
            preprocessed_images = tf.concat(
                [model.preprocess(image_tensor)[0]
                for image_tensor in image_tensors], axis=0)
            prediction_dict = model.predict(preprocessed_images, shapes)

            print("=================================================================")
            print(f"model: {model}\n")
            print(f"preprocessed_images: {preprocessed_images}\n")
            print(f"shapes: {shapes}\n")
            print(f"prediction_dict: {prediction_dict}\n")

            losses_dict = model.loss(prediction_dict, shapes)
            print(f"losses_dict: {losses_dict}\n")            
            
            total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
            gradients = tape.gradient(total_loss, vars_to_fine_tune)
            optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
        return total_loss

    return train_step_fn
