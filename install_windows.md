# FruitFinder v3

An AI fruit scanner for supermarket checkouts

* Should probs do a complete re-think of middle model / training code using the eager_few_shot_od_training_tf2_colab.ipynb code as a base
* Write model builder (/ loading from file) code (model.py)
* Writing training loop

---

## Usage Guide (WINDOWS)

### Installation

Follow [this guide](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html).

- Make sure to download [Microsoft C++ build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) if pycoco fails to install

- Do `python -m pip install .` instead of `python -m pip install --use-feature=2020-resolver .`

- If at the very last step of testing your installation you get XXX error, follow [this fix](https://stackoverflow.com/questions/71759248/importerror-cannot-import-name-builder-from-google-protobuf-internal) (see summary of fix below)

> Install the latest protobuf (in my case is 4.21.1)
> pip install --upgrade protobuf
> 
> Copy builder.py from ...\Lib\site-packages\google\protobuf\internal to your computer (let's say 'Documents')
> Install protobuf that compatible to your project, (for fruitfinder is 3.19.6)
> pip install protobuf==3.19.6
> 
> Copy builder.py from (let's say 'Documents') to Lib\site-packages\google\protobuf\internal
> run your code

---

<center> Made with ❤️ </center>
