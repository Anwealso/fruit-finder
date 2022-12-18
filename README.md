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

## Usage Guide v2

### Installation

Note: We are broadly following [this guide](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html), but with modifications along the way for our secific systerm conditions (e.g. M1 Mac)

#### Install Miniconda following [Apple's M1 guide](https://developer.apple.com/metal/tensorflow-plugin/)

[Download Conda environment](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh)

#### Create a new Anaconda virtual environment and activate it

```bash
conda create -n ff pip python=3.9
conda activate ff
```

#### Install tensorflow following [Apple's M1 guide](https://developer.apple.com/metal/tensorflow-plugin/)

```bash
conda install -c apple tensorflow-deps
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal
```

#### Verify the installation

This one will probably cause an error due to mismatched version of numpy (if so, fix in next step):

```bash
python verify_tf_install.py
```

Additionally test this one just to make sure:

```bash
python verify_tf_install_2.py
```

#### Fix the numpy issue

Following [this community fix](https://github.com/freqtrade/freqtrade/issues/4281), run:

```bash
pip install numpy --upgrade
```

#### Verify the Installation (Round 2)

Once again try to verify the installation with:

```bash
python verify_tf_install.py
```

You should see an outut something like this:

```text
Downloading data from https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
169001437/169001437 [==============================] - 44s 0us/step
Metal device set to: Apple M1 Pro

systemMemory: 16.00 GB
maxCacheSize: 5.33 GB

2022-11-26 09:24:32.456765: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:306] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2022-11-26 09:24:32.457057: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:272] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
2022-11-26 09:24:33.712254: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz
Epoch 1/5
/Users/alexnicholson/opt/miniforge3/envs/ff/lib/python3.9/site-packages/keras/backend.py:5582: UserWarning: "`sparse_categorical_crossentropy` received `from_logits=True`, but the `output` argument was produced by a Softmax activation and thus does not represent logits. Was this intended?
  output, from_logits = _get_logits(
2022-11-26 09:24:36.190124: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
782/782 [==============================] - 77s 88ms/step - loss: 4.9021 - accuracy: 0.0603   
Epoch 2/5
782/782 [==============================] - 67s 86ms/step - loss: 4.2499 - accuracy: 0.1134
Epoch 3/5
782/782 [==============================] - 67s 86ms/step - loss: 3.9722 - accuracy: 0.1417
Epoch 4/5
782/782 [==============================] - 68s 87ms/step - loss: 3.5762 - accuracy: 0.1880
Epoch 5/5
782/782 [==============================] - 66s 85ms/step - loss: 3.5478 - accuracy: 0.1934
```

Additionally double verify the installation with:

```bash
python verify_tf_install_2.py
```

You should see an output something like this:

```text
Metal device set to: Apple M1 Pro

systemMemory: 16.00 GB
maxCacheSize: 5.33 GB

2022-11-26 09:33:30.187567: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:306] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2022-11-26 09:33:30.187940: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:272] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
tf.Tensor(1497.3958, shape=(), dtype=float32)
```

Yay! Hopefully those all worked and now base tensorflow is fully installed. Next we're onto installing the Object Detection API...

#### Install the Tensorflow Object Detection API

##### Download the Object Detection API
Clone the [Object Detection API repo](https://github.com/tensorflow/models.git) into ./lib/models/.

```bash
git clone https://github.com/tensorflow/models.git ./lib/models
```

##### Protobuf Installation/Compilation

The Tensorflow Object Detection API uses Protobufs to configure model and training parameters. Before the framework can be used, the Protobuf libraries must be downloaded and compiled.

This should be done as follows:

* Head to the [protoc releases page](https://github.com/google/protobuf/releases)
* Download the latest protoc-*-*.zip release (e.g. protoc-3.12.3-win64.zip for 64-bit Windows)
* Extract the contents of the downloaded protoc-*-*.zip in a directory <PATH_TO_PB> of your choice (e.g. C:\Program Files\Google Protobuf)
* Add <PATH_TO_PB>\bin to your Path environment variable (see Environment Setup)
* In a new Terminal 1, cd into TensorFlow/models/research/ directory and run the following command:

``` bash
# From within models/research/
protoc object_detection/protos/*.proto --python_out=.
```

##### Install the Object Detection API

Installation of the Object Detection API is achieved by installing the object_detection package. This is done by running the following commands from within Tensorflow\models\research:

```bash
# From within models/research/
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

########################################################################################################

This should return a big error that tensorflow_io is broken like this:

```text
ERROR: Could not find a version that satisfies the requirement tensorflow_io (from object-detection) (from versions: none)
ERROR: No matching distribution found for tensorflow_io
```

To fix this, follow [this guide](https://developer.apple.com/forums/thread/688336) which says install tensorflow-io following [this other guide](https://stackoverflow.com/questions/70277737/cant-install-tensorflow-io-on-m1):

First, clone the tensorflow/io repository.

```bash
# From inside .../models/research/
git clone https://github.com/tensorflow/io.git
```

Then build it as shown below.

```bash
# From inside .../io/
python setup.py -q bdist_wheel
```

The wheel file will be created in the dist directory. You can then install the wheel by doing the following.

```bash
# From inside .../io/
python -m pip install --no-deps dist/tensorflow_io-0.28.0-cp39-cp39-macosx_11_0_arm64.whl
```

Now also install tf_slim manually.

```bash
pip install tf_slim
```

Now re-attempt to install object detection, now that we have the tensorflow-io depencancy installed

```bash
# From inside .../models/research/
python -m pip install --force --no-dependencies .
```

Still broken...

Try installing tensorflow-text manually

##### Test the installation

```bash
# From within .../models/research/
python object_detection/builders/model_builder_tf2_test.py
```





########################################################################################################






```bash
python -m pip install --no-deps .
```

And then install all the other deps of object_detection manually

```bash
pip install avro-python3 apache-beam pillow lxml matplotlib Cython contextlib2 pycocotools lvis scipy pandas 
pip install sacrebleu==2.2.0 pyparsing==2.4.7

```

Still not happy...

```bash
conda install tensorflow
pip install tf-models-official

```






##### Test your Installation

To test the installation, run the following command from within Tensorflow\models\research:

```bash
# From within /models/research/
python object_detection/builders/model_builder_tf2_test.py
```

---

## Usage Guide v1 (M1 Pro Mac)

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

### Training and Running Inference

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

## Data Labelling with LabelImg

### Installing LabelImg (for MacOS + Anaconda)

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
