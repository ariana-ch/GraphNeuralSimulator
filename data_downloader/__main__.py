import os
import sys
import shutil
from .download import download_dataset
from .convert import convert_dataset


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m data_loader <DATASET_NAME>")
        print("Example: python -m data_loader WaterDrop")
        sys.exit(1)

    dataset_name = sys.argv[1]
    tmp_dir = './tmp'

    try:
        # Step 1: Download the dataset
        download_dataset(dataset_name)

        # Step 2: Convert the dataset
        convert_dataset(os.path.join(tmp_dir, dataset_name))
    finally:
        # Step 3: Ensure the temporary directory is removed
        if os.path.exists(tmp_dir):
            print(f"Cleaning up temporary directory: {tmp_dir}")
            shutil.rmtree(tmp_dir)  # Remove the directory and its contents
        else:
            print(f"Temporary directory {tmp_dir} not found.")

        # Double-check and remove the directory if it persists (failsafe)
        try:
            if os.path.exists(tmp_dir):
                os.rmdir(tmp_dir)  # Attempt to remove empty directory
                print(f"Successfully removed {tmp_dir}")
        except OSError as e:
            print(f"Error removing {tmp_dir}: {e}")


if __name__ == '__main__':
    main()