import urllib.request
import zipfile
import os

url = "https://github.com/towhee-io/examples/releases/download/data/reverse_image_search.zip"
zip_path = "reverse_image_search.zip"
extract_path = "images_folder"

print(" Downloading dataset...")
urllib.request.urlretrieve(url, zip_path)

os.makedirs(extract_path, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print(f" Extraction complete! Files are in '{extract_path}'")