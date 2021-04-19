import io
import json

import albumentations
from albumentations import pytorch as AT
import torch

from PIL import Image

# Flask utils
from flask import Flask, request, render_template

# Define a flask app
from torch import nn
from torchvision import models
from torchvision.transforms import transforms

app = Flask(__name__)

model = models.resnext101_32x8d()  # we do not specify pretrained=True, i.e. do not load default weights

# replace the last fc layer with an untrained one (requires grad by default)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 196)
# Load trained model
model.load_state_dict(torch.load("resnext101_32x8d.pth"))
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([

        transforms.Resize(400),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


class_index = json.load(open('meta.json'))


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return class_index[predicted_idx]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        return get_prediction(image_bytes=img_bytes)


if __name__ == '__main__':
    app.run(debug=True)
