from io import BytesIO
from urllib import request
import onnxruntime as ort
from PIL import Image
import numpy as np

classifier_path = "hair_classifier_empty.onnx"


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.NEAREST)
    return img


# ImageNet normalization constants
MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
STD = np.array([0.229, 0.224, 0.225], dtype="float32")


def train_transform_small(img):
    img_array = (
        np.array(img).astype("float32") / 255.0
    )  # PIL to NumPy, normalize to [0, 1]
    img_array = (img_array - MEAN) / STD  # Normalize with ImageNet stats
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW (like ToTensor)
    return img_array


def lambda_handler(event, context):
    url = event["url"]

    img_raw = download_image(url)
    img = prepare_image(img_raw, [200, 200])
    img_array = train_transform_small(img)
    input_data = np.expand_dims(img_array, 0)

    sess = ort.InferenceSession(classifier_path)
    result = sess.run(None, {"input": input_data})
    body = result[0].tolist()[0][0]
    return {"statusCode": 200, "body": body}
