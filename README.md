# FruitFinder v3

An AI fruit scanner for supermarket checkouts

## Project Overview

### Desired Functionality

- Scan fruit
- Beep sound when recognised
- Show picture and name of predicted fruit (and maybe similar altermalives too) for user to confirm
- User confirms
- Weigh the fruit and add to cart
- Repeat

### Development Roadmap

- Get fruit images
- **[DONE]** Label with label.py & store imgs & labels together on file
- **[DONE]** Load training dataset from file w/ dataset.py load_data()
- **[DONE]** Load test dataset from file w/ dataset.py load_data()
- Finetute a fast & light object detection algorithm pretrained on ImageNet (Fast RCNN, YOLO, etc.) and save to file w/ train. py
- **[DONE] Load model & run live inference on webcam w/ predict. py**
- After validating performance on laptop webcam, compress and convert to tflite to be run on a raspberry pi + webcam
- Build GUI user interface with sounds and mount raspberry pi for a realistic scanning demo

### TODO:

- Update the data loading pipeline to allow for mulitple detection boxes per training image
- Write model builder (/ loading from file) code (model.py)
- Writing training loop

- Fix these export bugs where it says the model is not compiled properly (maybe we need to run 1 round of dummy inference first to init?) e.g.: `WARNING:tensorflow:Skipping full serialization of Keras layer <keras.layers.core.lambda_layer.Lambda object at 0x000001CF085B4FD0>, because it is not built.`

- Check that datatypes / detection box formatas are correct on the predivcted images benig plotted
- Add checkpoint saving for resuming from interrupted training
- Add tensorboard stats tracking
- Test running inference on pc with static images (using predict.py)

## Usage Guide

### Installation

See included installation guide for relevant platform.

### Training and Running Inference

- Activate the conda environment by running `conda activate fruitfinder`
- Run `python train.py` to train the model
- Run `python predict.py` to test out the trained model

## Finding Pretrained Models

- Look on the [TF Model Hub](https://tfhub.dev/s?module-type=image-object-detection&tf-version=tf2)
- Or on [TF2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

## Data Labelling (Label Studio <- new way)

Check out [their website](https://labelstud.io/)

Install with:

```bash
# Install the package into python virtual environment
pip install -U label-studio
# Launch it!
label-studio
```

## Data Labelling (LabelImg <- old way)

First clone the labelImg repo into ./lib/

```bash
git clone https://github.com/heartexlabs/labelImg.git
```

### Installing LabelImg (for MacOS + Anaconda)

```bash
conda install pyqt=5
conda install lxml
make qt5py3
```

From inside labelImg folder:

```
python -m pip install .
```

### Installing LabelImg (for Windows + Anaconda)

```bash
conda install pyqt=5;
conda install -c anaconda lxml;
pyrcc5 -o libs/resources.py resources.qrc;
```

Then, from inside labelImg folder:

```
python -m pip install .
```

See guide on [labelImg GitHub](https://github.com/heartexlabs/labelImg) for more installation options

### Using LabelImg

```bash
python labelImg.py
python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

## Other Resources

- https://github.com/tensorflow/models/issues/10499
- https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/
- https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

Model Exporting Methods

Method 1: Use the command line export script
Use `python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\ssd_mobilenet_v2_fpnlite_640x640_totoro\pipeline.config --trained_checkpoint_dir .\models\ssd_mobilenet_v2_fpnlite_640x640_totoro\checkpoint --output_directory .\exported-models\my_model`

- https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html?highlight=save#configure-the-training-pipeline
- https://medium.com/mlearning-ai/tensoflow-object-detection-api-with-tf1-vs-tf2-9d716be1f5d9

Method 2: Custom export just the frozen graph .pb

- https://leimao.github.io/blog/Save-Load-Inference-From-TF2-Frozen-Graph/

Method 3: Hack engineer a solution using exporter.py

- https://leimao.github.io/blog/Save-Load-Inference-From-TF2-Frozen-Graph/

---

<div align="center"> Made with ❤️<div>
