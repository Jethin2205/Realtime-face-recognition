import urllib.request
import os

os.makedirs("models", exist_ok=True)

models = [
    (
        "age_deploy.prototxt",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    ),
    (
        "age_net.caffemodel",
        "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/age_net.caffemodel"
    ),
    (
        "gender_deploy.prototxt",
        "https://raw.githubusercontent.com/eveningglow/age-and-gender-classification/master/model/gender_deploy.prototxt"
    ),
    (
        "gender_net.caffemodel",
        "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/gender_net.caffemodel"
    ),
]

for filename, url in models:
    path = os.path.join("models", filename)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024*1024)
        if size_mb > 1:
            print(f"Already exists: {filename} ({size_mb:.0f} MB)")
            continue
        else:
            os.remove(path)

    print(f"Downloading {filename} ...")
    try:
        urllib.request.urlretrieve(url, path)
        size_mb = os.path.getsize(path) / (1024*1024)
        print(f"Done: {filename} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"Failed: {filename} → {e}")

print("\nDone! Run: python main.py")
