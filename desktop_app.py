from tkinter import *
from PIL import ImageTk, Image
# import os
import time

index = 0
path = "data/totoro/images/test/dad_{}.jpg"

import tkinter as tk

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


# ----------------------------- SETUP THE WINDOW ----------------------------- #
root = tk.Tk()
root.geometry("1000x1000")
root.resizable(width=True, height=True)
myapp = App(root)


# ----------------------------------- MAIN ----------------------------------- #
myapp.update()

while True:
    time.sleep(1)
    myapp.my_num.set(myapp.my_num.get()+1)
    myapp.text_box.config(text=myapp.my_num.get())
    myapp.update()
    
myapp.mainloop()