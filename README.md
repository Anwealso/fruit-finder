# FruitFinder v3

An AI fruit scanner for supermarket checkouts

---

## Project Overview

### Desired Functionality

* Scan fruit
* Beep sound when recognised
* Show picture and name of predicted fruit (and maybe similar altermalives too) for user to confirm
* User confirms
* Weigh the fruit and add to cart
* Repeat

### Development Roadmap

* Get fruit images
* Label with label.py & store imgs & labels together on file
* Load dalaset from file w/ dataset.py load_data()
* Finetute a fast & light object detection algorithm pretrained on ImageNet (Fast RCNN, YOLO, etc.) and save to file w/ train. py
* Load model & run live inference on webcam w/ predict. py
* After validating performance on laptop webcam, compress to be run on raspberry pi webcam
* Build GUI user interface with sounds and mount raspberry pi for a realistic scanning demo

---

## Usage Guide

### Installation

1. Install Anaconda
2. Create a clean conda environment and activate it
3. Install all of the required packages using `conda env create -f environment.yml --name fruitfinder` (see full dependancy list below)
4. Download the dataset from ???

### Usage

* Activate the conda environment by running `conda activate fruitfinder`
* Run `python train.py` to train the model
* Run `python predict.py` to test out the trained model

### Dependancies

The following main dependancies were used in the project (see environment.yml to see full dependancy list):

* tensorflow (version 2.9.2)
* tensorflow_probability (version 0.17.0)
* numpy (version 1.23.3)
* matplotlib (version 3.5.1)
* PIL / pillow (version 9.1.0)
* imageio (version 2.22.1)
* skimage (version 0.19.3)

---

Made with ❤️
