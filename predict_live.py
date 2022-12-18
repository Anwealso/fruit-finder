"""
predict.py

30/10/2022

Shows example usage of the trained model on live webcam video with detection boxes shown overlayed

"""


import dataset
import utils
from tensorflow import keras
import tensorflow as tf


if __name__ == "__main__":
    # ---------------------------------------------------------------------------- #
    #                                  PARAMETERS                                  #
    # ---------------------------------------------------------------------------- #
    LABELS_PATH = ".\\data\\alex\\mscoco_label_map.pbtxt"
    MODEL_PATH = ".\\models\\ssd_mobilenet_v2_2"

    # ---------------------------------------------------------------------------- #
    #                                   LOAD DATA                                  #
    # ---------------------------------------------------------------------------- #
    # Import data loader from dataset.py
    # (train_data, test_data, label_map) = dataset.load_dataset(max_images=None, verbose=True)


    # ---------------------------------------------------------------------------- #
    #                             IMPORT TRAINED MODEL                             #
    # ---------------------------------------------------------------------------- #
    # Import trained and saved model from file
    print("Loading model ...")
    # model = tf.saved_model.load(".\\models\\ssd_resnet50_v1_fpn_640x640_coco17_tpu-8\\saved_model")
    model = tf.saved_model.load(MODEL_PATH)


    # ---------------------------------------------------------------------------- #
    #                                 FINAL RESULTS                                #
    # ---------------------------------------------------------------------------- #

    # Visualise the live detection capability on the webcam
    # utils.view_webcam()
    print("Running inference on webcam ...")
    utils.run_detector_live(model, LABELS_PATH)