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

See included installation guide for relevant platform.

---

## Finding Pretrained Models

- Look on the [TF Model Hub](https://tfhub.dev/s?module-type=image-object-detection&tf-version=tf2)

## Data Labelling

### Installing LabelImg

#### MacOS Anaconda

```bash
conda install pyqt=5
conda install lxml
make qt5py3
```

#### Windows Anaconda

```bash
conda install pyqt=5;
conda install -c anaconda lxml;
pyrcc5 -o libs/resources.py resources.qrc;
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
