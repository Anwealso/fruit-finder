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


def load_dataset(max_images=None, verbose=False):
    """
    Loads the OASIS dataset of brain MRI images

        Parameters:
            (optional) max_images (int): The maximum number of images of the dataset to be used (default=None)
            (optional) verbose (bool): Whether a description of the dataset should be printed after it has loaded

        Returns:
            train_data_scaled (ndarray): Numpy array of scaled image data for training (9,664 images max)
            test_dat_scaleda (ndarray): Numpy array of scaled image data testing (1,120 images max)
            validate_data_scaled (ndarray): Numpy array of scaled image data validation (544 images max)
            data_variance (int): Variance of the test dataset
    """

    print("Loading dataset...")

    # File paths
    images_path = "keras_png_slices_data/"
    test_path = images_path + "keras_png_slices_test/"
    train_path = images_path + "keras_png_slices_train/"
    validate_path = images_path + "keras_png_slices_validate/"
    dataset_paths = [test_path, train_path, validate_path]

    # Set up the lists we will load our data into
    test_data = []
    train_data = []
    validate_data = []
    datasets = [test_data, train_data, validate_data]

    # Load all the images into numpy arrays
    for i in range(0, len(dataset_paths)):
        # Get all the png files in this dataset_path directory
        images_list = glob.glob(os.path.join(dataset_paths[i], "*.png"))

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
    # Scale the data into values between -0.5 and 0.5 (range of 1 centred about 0)
    train_data_scaled = (train_data / 255.0) - 0.5
    test_data_scaled = (test_data / 255.0) - 0.5
    validate_data_scaled = (validate_data / 255.0) - 0.5

    # Get the dataset variance
    data_variance = np.var(train_data / 255.0)

    if verbose == True:
        # Debug dataset loading    
        print(f"###train_data ({type(train_data)}): {np.shape(train_data)}###")
        print(f"###test_data ({type(test_data)}): {np.shape(test_data)}###")
        print(f"###train_data_scaled ({type(train_data_scaled)}): {np.shape(train_data_scaled)}###")
        print(f"###test_data_scaled ({type(test_data_scaled)}): {np.shape(test_data_scaled)}###")
        print(f"###data_variance ({type(data_variance)}): {data_variance}###")
        print('')

        print(f"###validate_data ({type(validate_data)}): {np.shape(validate_data)}###")
        print(f"###validate_data_scaled ({type(validate_data_scaled)}): {np.shape(validate_data_scaled)}###")

        print('')
        print('')

    return (train_data_scaled, validate_data_scaled, test_data_scaled, data_variance)


def get_labels():
    # TODO: Document me!
    # Return the COCO Label Map
    return {
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


if __name__ == "__main__":
    # Run a test
    load_dataset(max_images=1000)