from tkinter import *
from PIL import ImageTk, Image
import time
import tkinter as tk
import tensorflow as tf
import utils
import cv2
import numpy as np
import dataset

# --------------------------------- CONSTANTS -------------------------------- #
path = "data/totoro/images/test/dad_{}.jpg"

DATASET_PATH = "data/totoro/"
LABELS_PATH = DATASET_PATH + "label_map.pbtxt"
MODEL_PATH = "./exported-models/my_model/saved_model"


def run_inference(model, labels_file, min_score=0.5):
    # Run inference on image
    result = model(img)
    result = {key:value.numpy() for key,value in result.items()}

    # Convert Class Indices to Class Names
    labels_dict = dataset.get_tf_labels_from_file(labels_file)
    class_names = [list(labels_dict.keys())[list(labels_dict.values()).index(int(item))] for item in result["detection_classes"][0]]

    # If there is a detection, play a beep
    for score in result["detection_scores"][0]:
        if (score > min_score):
            print("DETECTED!")
            # play_beep()
            break

    image_with_boxes = utils.draw_boxes(
        np.squeeze(img), 
        result["detection_boxes"],
        class_names,
        result["detection_scores"][0],
        max_boxes=10,
        min_score=min_score)

    return image_with_boxes


class App(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(side = "bottom", fill = "both", expand = "yes")

        # --------------------------------- VARIABLES -------------------------------- #
        self.my_num = tk.IntVar()
        self.my_num.set(1)


        # -------------------------------- COMPONENTS -------------------------------- #
        # # Input box
        # self.entrythingy = tk.Entry()
        # self.entrythingy.pack()

        # # Yellow text box
        self.text_box = Label(self, text = "Hello, World!", bg = "yellow", height = 10, width = 15, relief = "solid", cursor = "target")  
        self.text_box.pack()

        # Image
        img = ImageTk.PhotoImage(Image.open(path.format(self.my_num.get())), height = 10, width = 15)
        self.image_section = Label(self, image = img)
        self.image_section.image = img
        self.image_section.pack()


        # --------------------------------- CALLBACKS -------------------------------- #
        # Define a callback for when the user hits return.
        # It prints the current value of the variable.
        master.bind('<Key-Return>',self.print_contents)


    def print_contents(self, event):        
        print(f"My Number {self.my_num.get()}")

        self.my_num.set(self.my_num.get()+1)

        new_img = ImageTk.PhotoImage(Image.open(path.format(self.my_num.get())))
        self.image_section.configure(image=new_img)
        self.image_section.image = new_img


# ------------------------------ SETUP THE MODEL ----------------------------- #
# Import trained and saved model from file
print("Loading model ...")
# model = tf.saved_model.load(".\\models\\ssd_resnet50_v1_fpn_640x640_coco17_tpu-8\\saved_model")
model = tf.saved_model.load(MODEL_PATH)

print("Connecting to webcam ...")
# Define a video capture object
vid = cv2.VideoCapture(0)

# ----------------------------- SETUP THE WINDOW ----------------------------- #
root = tk.Tk()
root.geometry("1000x1000")
root.resizable(width=True, height=True)
myapp = App(root)


# ----------------------------------- MAIN ----------------------------------- #
myapp.update()

while True:
    # Capture the video frame by frame
    ret, frame = vid.read()
    img = np.expand_dims(frame, 0)

    processed_image = run_inference(model, LABELS_PATH, min_score=0.5)

    myapp.image_section.configure(image=processed_image)
    myapp.image_section.image = processed_image

    time.sleep(1)
    # myapp.my_num.set(myapp.my_num.get()+1)
    # myapp.text_box.config(text=myapp.my_num.get())
    
    myapp.update()

myapp.mainloop()