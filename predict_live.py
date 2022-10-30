"""
predict.py

30/10/2022

Shows example usage of the trained model on live webcam video with detection boxes shown overlayed

"""


import dataset
import utils
from tensorflow import keras


if __name__ == "__main__":
    # ---------------------------------------------------------------------------- #
    #                                   LOAD DATA                                  #
    # ---------------------------------------------------------------------------- #
    # Import data loader from dataset.py
    # (train_data, test_data, label_map) = dataset.load_dataset(max_images=None, verbose=True)


    # ---------------------------------------------------------------------------- #
    #                             IMPORT TRAINED MODEL                             #
    # ---------------------------------------------------------------------------- #
    # Import trained and saved vqvae model from file
    # trained_model = keras.models.load_model("./saved_model")


    # ---------------------------------------------------------------------------- #
    #                                 FINAL RESULTS                                #
    # ---------------------------------------------------------------------------- #
    utils.view_webcam()

    # Visualise the live detection capability on the webcam
    # utils.show_detection_webcam(trained_model)

