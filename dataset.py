"""
dataset.py

Alex Nicholson (45316207)
11/10/2022

Contains the data loader for loading and preprocessing the OASIS data

TODO: Fix documentation

"""


import glob
import numpy as np
import PIL
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import utils as utils
from lib.object_detection.utils import label_map_util
# TODO: fix thes messed up path importing issue


def plot_detections(image_np, boxes, classes, scores, category_index, figsize=(12, 16), image_name=None):

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

    utils.draw_boxes(
        image_np_with_annotations,
        boxes,
        classes,
        scores,
        # category_index,
        # use_normalized_coordinates=True,
        min_score=0.8)

    if image_name:
        plt.imsave(image_name, image_np_with_annotations)
    else:
        plt.imshow(image_np_with_annotations)



def get_tf_labels_from_file(path):
    """
    Gets the labels map as a dict of name:id

        Parameters:
            path (str): the path to the labels map file

        Returns:
            label_map (dict): Dict of class name:id pairs
    """

    return label_map_util.get_label_map_dict(path, use_display_name=True)

def write_list_to_file(list, filename):
    """
    TODO: Document me!

    """
    with open(filename, 'w') as fp:
        for item in list:
            # write each item on a new line
            fp.write("%s\n" % item)


def annotate_dataset(dataset_path, images_folder_path, annotations_folder_path):
    
    # If annotations dont already exist, make them now...
    tf_classes_path = dataset_path + "mscoco_label_map.pbtxt"
    labelimg_classes_path = dataset_path + "labelImg_classes.txt"

    # Create a labelImg-format class labels file from our label_map.pbtxt
    labels_list = list(get_tf_labels_from_file(tf_classes_path).keys())
    write_list_to_file(labels_list, labelimg_classes_path)
    
    # Run labelImg utility
    print("Opening labelImg ...")
    print(f"python ./lib/labelImg/labelImg.py {images_folder_path} {labelimg_classes_path} {annotations_folder_path}")
    os.system(f"python ./lib/labelImg/labelImg.py {images_folder_path} {labelimg_classes_path} {annotations_folder_path}")


def xml2csv(location, dataset_segment):
    """
    Gets the labels map as a dict of name:id

        Parameters:
            location (str): a path to the data labels for images of a specific 
                class and dataset segment, e.g. DATASETNAME/labels/train/CLASSNAME/
            dataset_segment (str): the segment of the dataset we are 
                processing (train, validate, or test)

        Returns:
            label_map (dict): Dict of class name:id pairs
    """
    
    # print("NOW INSIDE xml2csv...")
    # print(location)

    # To parse the xml files
    import xml.etree.ElementTree as ET

    # Return list
    temp_res = []

    # Run through all the files
    for file in os.listdir(location):
        # print(location)

        # Check the file name ends with xml
        if file.endswith(".xml"):

            # Get the file name
            file_whole_name = f"{location}/{file}"

            # Open the xml name
            tree = ET.parse(file_whole_name)
            root = tree.getroot()

            # Get the width, height of images
            #  to normalize the bounding boxes
            size = root.find("size")
            width, height = float(size.find("width").text), float(size.find("height").text)

            # Find all the bounding objects
            for label_object in root.findall("object"):
                # Make temp array for csv...

                # The dataset segment type
                temp_csv = [str(dataset_segment)]

                # Image path
                image_location = location.split("/")
                image_location[-3] = "images"
                image_location = "/".join(image_location)
                image_path = f"{image_location}/{os.path.splitext(file)[0]}.jpg"
                temp_csv.append(image_path)

                # Class label
                temp_csv.append(label_object.find("name").text)

                # Bounding box coordinate
                bounding_box = label_object.find("bndbox")

                # Add the upper left coordinate
                x_min = float(bounding_box.find("xmin").text) / width
                y_min = float(bounding_box.find("ymin").text) / height
                temp_csv.extend([x_min, y_min])

                # Add the lower left coordinate (not necessary, left blank)
                # temp_csv.extend(["", ""])

                # Add the lower right coordinate
                x_max = float(bounding_box.find("xmax").text) / width
                y_max = float(bounding_box.find("ymax").text) / height
                temp_csv.extend([x_max, y_max])

                # Add the upper right coordinate (not necessary, left blank)
                # temp_csv.extend(["", ""])

                # Append to the res
                temp_res.append(temp_csv)

    return temp_res


def load_labels_from_file(location):
    """
    Loads the labels for each of the images (box size, position and class) in each of the train, validate and test datasets
    
        Parameters:
            location (str): the path to the base labels folder (labels/)

        Returns:
            res (pd.Dataframe): The labels
    """

    # Array for final csv file
    res = []

    # Get all the file in dir
    for training_type_dir in os.listdir(location):
        # Get the dirname
        dir_name = f"{location}/{training_type_dir}"
        # Check whether is dir
        if os.path.isdir(dir_name):
            # Process the files
            for class_name in os.listdir(dir_name):
                # Check whether this file is dir, if so dont process it
                if os.path.isdir(f"{dir_name}/{class_name}"):
                    # Convert the chosen extension to csv
                    res.extend(xml2csv(f"{dir_name}/{class_name}", training_type_dir))

    # Get the result as a dataframe to return
    res_csv = pd.DataFrame(res,
                           columns=["set", "path", "label",
                                    "x_min", "y_min",
                                    "x_max", "y_max"])
    
    return res_csv


def get_annotations(dataset_path):
    """
    Loads the annotations from file or prompts the user to complete the annotations if not completed.
    
        Parameters:
            location (str): the path to the dataset folder

        Returns:
            res (pd.Dataframe): The labels
    """

    gt_boxes = []

    train_images_subfolders = glob.glob(dataset_path + "images/train/"+  "*")
    class_names = list()

    for subfolder in train_images_subfolders:
        class_names.append(subfolder.split("/")[-1])

    for class_name in class_names:
        # Do the annotation now if not already done
        training_num_images = len(glob.glob(dataset_path + f"images/train/{class_name}/"+  "*"))
        training_num_labels = len(glob.glob(dataset_path + f"labels/train/{class_name}/"+  "*"))

        if training_num_labels != training_num_images:
            # If training annotations dont already exist, make them now...
            annotate_dataset(dataset_path, dataset_path + f"images/train/{class_name}", dataset_path + f"labels/train/{class_name}")
        else:
            print(f"Training data already annotated for class: {class_name}")

    # Load the annotations from file
    gt_boxes = load_labels_from_file(dataset_path + "labels")

    return gt_boxes


def load_dataset(dataset_path, max_images=None, verbose=False):
    """
    Loads the dataset at he given path

        Parameters:
            (optional) max_images (int): The maximum number of images of the dataset to be used (default=None)
            (optional) verbose (bool): Whether a description of the dataset should be printed after it has loaded

        Returns:
            datasets (tuple): A tuple containing the three datasets (train_dataset, validate_dataset, test_dataset). Each dataset is a dictionary of {'images': <ndarray>, 'labels': <list>}
    """

    print("Loading dataset...")

    # File paths
    # TODO: Update this to work for multiple classes (multiple class image sub folders)
    test_path = dataset_path + "images/test/duck"
    train_path = dataset_path + "images/train/duck"
    # validate_path = images_path + "validate/"
    dataset_paths = [test_path, train_path]#, validate_path]

    # Set up the lists we will load our data into
    test_data = []
    train_data = []
    # validate_data = []
    datasets = [test_data, train_data]#, validate_data]
    print(train_data)

    # ------------------------------ Load the images ----------------------------- #

    # Load all the images into numpy arrays
    for i in range(0, len(dataset_paths)):
        # Get all the files in this dataset_path directory that arent .xml fils (all the images)
        images_list = glob.glob(os.path.join(dataset_paths[i], "*.[!xml]*"))

        images_collected = 0 
        for img_filename in images_list:
            # Stop loading in images if we hit out max image limit
            if max_images and images_collected >= max_images:
                break

            # Open the image
            img = PIL.Image.open(img_filename)
            # Convert image to numpy array
            data = np.asarray(img)
            datasets[i].append(data)

            # Close the image (not strictly necessary)
            del img
            images_collected = images_collected + 1
            
    # Convert the datasets into numpy arrays
    train_images_np = np.array(train_data)
    # test_images_np = np.array(test_data)
    # validate_data = np.array(validate_data)

    # ------------------------------ Load the labels ----------------------------- #

    # Load the training data labels from file or annotate them now
    annotations = get_annotations(dataset_path)

    # Convert to corner-coord-only train_gt_boxes format
    train_gt_boxes = list()
    for i in range(0, len(annotations)):
        row = annotations.loc[i, ['x_min', 'y_min', 'x_max', 'y_max']]
        box_coords_np = row.to_numpy(dtype=np.float32)
        train_gt_boxes.append(box_coords_np)


    # ------------------------- Prepare data for training ------------------------ #
    
    # TODO: Update this for mutliple class functionality
    
    # By convention, our non-background classes start counting at 1.  Given
    # that we will be predicting just one class, we will therefore assign it a
    # `class id` of 1.
    duck_class_id = 1
    num_classes = 1

    category_index = {duck_class_id: {'id': duck_class_id, 'name': 'rubber_ducky'}}

    # Convert class labels to one-hot; convert everything to tensors.
    # The `label_id_offset` here shifts all classes by a certain number of indices;
    # we do this here so that the model receives one-hot labels where non-background
    # classes start counting at the zeroth index.  This is ordinarily just handled
    # automatically in our training binaries, but we need to reproduce it here.

    label_id_offset = 1
    train_image_tensors = []
    gt_classes_one_hot_tensors = []
    gt_box_tensors = []

    for (train_image_np, gt_box_np) in zip(train_images_np, train_gt_boxes):
        train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(train_image_np, dtype=tf.float32), axis=0))

        gt_box_tensors.append(tf.convert_to_tensor(gt_box_np, dtype=tf.float32))

        zero_indexed_groundtruth_classes = tf.convert_to_tensor(np.ones(shape=[gt_box_np.shape[0]], dtype=np.int32) - label_id_offset)

        gt_classes_one_hot_tensors.append(tf.one_hot(zero_indexed_groundtruth_classes, num_classes))

    print('Done prepping data.')

    # ----------------------- visualize the rubber duckies ----------------------- #

    # Fix the fact that the ret boxes are plotting like garbage on the images
    if verbose == True:
        dummy_scores = np.array([1.0], dtype=np.float32)  # give boxes a score of 100%
        images_to_show = 5

        plt.figure(figsize=(12, 8))

        for idx in range(images_to_show):
            plt.subplot(2, 3, idx+1)

            # TODO: Need to rework this shit to work for more than 1 detection box per image
            plot_detections(
                train_images_np[idx],
                [train_gt_boxes[idx]],
                ["duck"],
                dummy_scores, 
                category_index)

        plt.show()

    # --------------------------------- OLD CODE --------------------------------- #
    if verbose == True:
        # Debug dataset loading
        print()
        print(f"Annotations:\n{annotations}")
        print()

        print(f"Training Data:")
        print(f"    Images:  ({type(train_images_np)})  {np.shape(train_images_np)}")
        print(f"    Labels:  ({type(train_gt_boxes)})  {np.shape(train_gt_boxes)}")

        # print(f"###test_data ({type(test_data)}): {np.shape(test_data)}###")
        # print(f"###validate_data ({type(validate_data)}): {np.shape(validate_data)}###")
        print('\n\n')


    # TODO: adapt this code so we can return the train, validate and test datasets
    return ({"images": train_images_np, "labels": train_gt_boxes}, {}, {})


if __name__ == "__main__":
    # Run a test
    load_dataset(max_images=1000)