import requests
import base64
import tensorflow as tf

api_url = "http://localhost:8400/"
filename = "data/totoro/images/test/dad_2.jpg"

# Read in and convert the image
with open(filename, "rb") as image_file:
    image_string = base64.b64encode(image_file.read())

# Headers
headers = {"content-type": "image/jpeg"}

# Send to the server
response = requests.post(api_url, data=image_string, headers=headers)

# Get the processed image back from the server
if response.ok:
    print(response.json())