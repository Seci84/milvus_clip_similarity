import urllib.request
import zipfile
import os


class DatasetDownloader:
    def __init__(self, url:str, zip_path: str = "dataset.zip", extract_path: str = "data"):
        self.url = url
        self.zip_path = zip_path
        self.extract_path = extract_path

    def download(self):
        urllib.request.urlretrieve(self.url, self.zip_path)

    def extract(self):
        os.makedirs(self.extract_path, exist_ok = True)
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.extract_path)

    def cleanup(self):
        if os.path.exists(self.zip_path):
            os.remove(self.zip_path)

    def run(self, clean_after:bool = True):
        self.download()
        self.extract()
        if clean_after:
            self.cleanup()

if __name__ == "__main__":
    url = "https://github.com/towhee-io/examples/releases/download/data/reverse_image_search.zip"
    downloader = DatasetDownloader(url, zip_path="reverse_image_search.zip", extract_path = "sample_data")
    downloader.run()
            
    