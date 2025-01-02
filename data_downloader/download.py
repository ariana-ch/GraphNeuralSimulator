import os
import sys

import requests


def download_file(url, output_path):
    """
    Download a file from a URL to a specified path.
    """
    try:
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
        sys.exit(1)


def download_dataset(dataset_name):
    """
    Download all necessary files for the specified dataset.
    """
    print(f'Downloading dataset: {dataset_name}')
    base_url = f"https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/{dataset_name}/"
    files = ["metadata.json", "train.tfrecord", "valid.tfrecord", "test.tfrecord"]

    # Ensure the output directory exists
    os.makedirs(f'./tmp/{dataset_name}', exist_ok=True)

    # Download each file
    for file_name in files:
        file_url = f"{base_url}{file_name}"
        file_path = os.path.join(f'./tmp/{dataset_name}', file_name)
        download_file(file_url, file_path)


if __name__ == "__main__":
    download_dataset('WaterDropSample')
    if len(sys.argv) != 3:
        print("Usage: python download_dataset.py <DATASET_NAME> <OUTPUT_DIR>")
        print("Example: python download_dataset.py WaterDrop /tmp/")
        sys.exit(1)

    dataset_name = sys.argv[1]
    output_dir = os.path.join(sys.argv[2], dataset_name)

    download_dataset(dataset_name)