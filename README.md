# FruitFinder v3

An AI fruit scanner for supermarket checkouts

* Should probs do a complete re-think of middle model / training code using the eager_few_shot_od_training_tf2_colab.ipynb code as a base
* Write model builder (/ loading from file) code (model.py)
* Writing training loop

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
* **[DONE] Label with label.py & store imgs & labels together on file**
* **[DONE]Load training dataset from file w/ dataset.py load_data()**
* Load test dataset from file w/ dataset.py load_data()
* Finetute a fast & light object detection algorithm pretrained on ImageNet (Fast RCNN, YOLO, etc.) and save to file w/ train. py
* **[DONE] Load model & run live inference on webcam w/ predict. py**
* After validating performance on laptop webcam, compress to be run on raspberry pi webcam
* Build GUI user interface with sounds and mount raspberry pi for a realistic scanning demo

TODO:

* Update the data loading pipeline to allow for mulitple detection boxes per training image

---

## Usage Guide

### Installation

1. Install Anaconda
2. Create a clean conda environment and activate it
3. Install all of the required packages using `conda env create -f environment.yml --name fruitfinder` (see full dependancy list below)
4. Download the resnet50 model from [here](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz). Note: various other COCO models accessible [here](https://tfhub.dev/tensorflow/collections/object_detection/1)

<!-- Methods of getting object_detection tools working:

1. (NOT WORKING) Add the object_detection library to the conda env path by doing the following:
![this guide](conda-path-guide.png)

2. (WORKING SO FAR) Maually add slim/ and object_detection/ into the site-packages of the conda env -->

### Installing the Object Detection API

```bash
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install --no-deps  .
```

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

## Data Labelling

### Installing LabelImg

#### MacOS Anaconda

```bash
conda install pyqt=5
conda install lxml
make qt5py3
```

See guide on [labelImg GitHub](https://github.com/heartexlabs/labelImg) for more installation options

### Using LabelImg

```bash
python labelImg.py
python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

---

## Other Resources

https://github.com/tensorflow/models/issues/10499

https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md



---

<center> Made with ❤️ </center>
