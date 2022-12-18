# FruitFinder v3

An AI fruit scanner for supermarket checkouts

* Should probs do a complete re-think of middle model / training code using the eager_few_shot_od_training_tf2_colab.ipynb code as a base
* Write model builder (/ loading from file) code (model.py)
* Writing training loop

---

## Usage Guide (MACOS M1 ARM)

### Installation (CURRENTLY BROKEN)

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

<center> Made with ❤️ </center>
