"""
predict.py

30/10/2022

Shows example usage of the trained model on individual example images with visualisations of it's output results

"""


import dataset
import utils
import tensorflow as tf


if __name__ == "__main__":
    # ---------------------------------------------------------------------------- #
    #                                  PARAMETERS                                  #
    # ---------------------------------------------------------------------------- #
    EXAMPLES_TO_SHOW = 10 # number of test detection examples to show

    DATASET_PATH = "data\\totoro\\"
    IMAGES_PATH = DATASET_PATH + "images\\test\\"
    OUTPUT_PATH = DATASET_PATH + "out\\test\\"
    LABELS_PATH = DATASET_PATH + "label_map.pbtxt"
    MODEL_PATH = ".\\exported-models\\my_model\\saved_model"

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
    # Visualise the final results and calculate the accuracy of the detections
    # utils.show_detection_examples(trained_model, test_data, EXAMPLES_TO_SHOW)

    print("Running inference on static images ...")
    utils.run_detector(model, IMAGES_PATH, OUTPUT_PATH, LABELS_PATH, min_score=0.2)