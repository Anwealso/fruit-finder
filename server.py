from flask import Flask, request, jsonify
import utils
import tensorflow as tf
from PIL import Image
import base64
import io
import numpy as np


# --------------------------------- VARIABLES -------------------------------- #
LABELS_PATH = "data\\totoro\\label_map.pbtxt"
MODEL_PATH = ".\\exported-models\\my_model\\saved_model"


# ----------------------------------- INIT ----------------------------------- #
app = Flask(__name__)
model = tf.saved_model.load(MODEL_PATH)


# ----------------------------------- MAIN ----------------------------------- #
@app.route("/", methods = ["POST"])
def index():
    if request.method == "POST":

        # Get the sent image
        stringified_img = request.data
        img = Image.open(io.BytesIO(base64.b64decode(stringified_img)))
        img.save('server_test/original_image.png', 'PNG')

        # Run inference
        processed_image = utils.run_detector_api(model, LABELS_PATH, img)

        # Convert back to PIL Image
        processed_image.save('server_test/processed_image.png', 'PNG')
        image_string = base64.b64encode(processed_image.tobytes())

        return jsonify({"processed_image": str(image_string)})
        # return image_string
         
    # else:
    #     return "<p>Hello, World!</p>"

if __name__ == "__main__":
    app.run(port=8400)