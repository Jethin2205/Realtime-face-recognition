import urllib.request
import os

os.makedirs("models", exist_ok=True)

models = [
    (
        "age_deploy.prototxt",
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt"
    ),
    (
        "age_net.caffemodel",
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_chalearn_iccv2015.caffemodel"
    ),
    (
        "gender_deploy.prototxt",
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt"
    ),
    (
        "gender_net.caffemodel",
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.caffemodel"
    ),
    (
        "emotion_mini_xception.onnx",
        "https://github.com/serengil/deepface/raw/master/deepface/models/facial_expression/facial_expression_model_weights.h5"
    ),
]

for filename, url in models:
    path = os.path.join("models", filename)
    if os.path.exists(path):
        print(f"Already exists: {filename}")
        continue
    print(f"Downloading {filename} ...")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"Done: {filename}")
    except Exception as e:
        print(f"Failed: {filename} -> {e}")
        print(f"Download manually from: {url}")

print("\nAll done! Check your models/ folder.")
