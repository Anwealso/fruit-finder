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
from object_detection.utils import label_map_util
# from object_detection.utils import colab_utils


# def get_training_labels(dataset_path, train_data):
#     """
#     TODO: Document me!!!

#     """

#     gt_boxes = []

#     if False:
#         # if annotations already finished
#         # load anotations
#         load_labels_from_file()

#     else:
#         # else annotate now
#         colab_utils.annotate(train_images_np, box_storage_pointer=gt_boxes)

#     return gt_boxes


def load_dataset(dataset_path, max_images=None, verbose=False):
    """
    Loads the OASIS dataset of brain MRI images

        Parameters:
            (optional) max_images (int): The maximum number of images of the dataset to be used (default=None)
            (optional) verbose (bool): Whether a description of the dataset should be printed after it has loaded

        Returns:
            train_data (ndarray): Numpy array of image data for training
            test_data (ndarray): Numpy array of image data testing
    """

    print("Loading dataset...")

    # File paths
    test_path = dataset_path + "test/"
    train_path = dataset_path + "train/"
    # validate_path = images_path + "validate/"
    dataset_paths = [test_path, train_path]#, validate_path]

    # Set up the lists we will load our data into
    test_data = []
    train_data = []
    # validate_data = []
    datasets = [test_data, train_data]#, validate_data]

    # Load all the images into numpy arrays
    for i in range(0, len(dataset_paths)):
        # Get all the png files in this dataset_path directory
        images_list = glob.glob(os.path.join(dataset_paths[i], "*.*"))

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
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    validate_data = np.array(validate_data)

    # Preprocess the data
    train_data = np.expand_dims(train_data, -1)
    test_data = np.expand_dims(test_data, -1)
    validate_data = np.expand_dims(validate_data, -1)

    # Load the training data labels from file or annotate them now
    # gt_boxes = get_training_labels(dataset_path, train_data)


    # # Scale the data into values between -0.5 and 0.5 (range of 1 centred about 0)
    # train_data_scaled = (train_data / 255.0) - 0.5
    # test_data_scaled = (test_data / 255.0) - 0.5
    # validate_data_scaled = (validate_data / 255.0) - 0.5

    # # Get the dataset variance
    # data_variance = np.var(train_data / 255.0)

    if verbose == True:
        # Debug dataset loading    
        print(f"###train_data ({type(train_data)}): {np.shape(train_data)}###")
        print(f"###test_data ({type(test_data)}): {np.shape(test_data)}###")
        # print(f"###validate_data ({type(validate_data)}): {np.shape(validate_data)}###")
        print('')
        print('')

    return (train_data, test_data)


def get_labels(path):
    """
    Gets the labels map as a dict of name:id

        Parameters:
            path (str): the path to the labels map file

        Returns:
            label_map (dict): Dict of class name:id pairs
    """

    return label_map_util.get_label_map_dict(path, use_display_name=True)

if __name__ == "__main__":
    # Run a test
    load_dataset(max_images=1000)