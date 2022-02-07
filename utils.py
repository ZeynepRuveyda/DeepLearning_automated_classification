# importing required libraries
import os
import pandas as pd
from pathlib import Path
from zipfile import ZipFile

# Creating csv files for all image with their classes. Classes will be added as folder of image name.
def create_csv_file(dir_zip):
    with ZipFile(dir_zip, 'r') as zipObj:
        # Get list of files names in zip
        file_paths = zipObj.namelist()
    # returning all file paths
    images_paths = []
    for f in file_paths:
        path = os.path.join(os.path.expanduser('~'), dir_zip, f)
        images_paths.append(path)

    names = []
    labels = []
    image_types = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']

    for image_path in images_paths:
        filename, file_extension = os.path.splitext(image_path)
        latest_file = Path(image_path).parent.absolute()
        latest_file_name = Path(latest_file).stem
        if file_extension in image_types:
            names.append(Path(image_path).stem)
            labels.append(latest_file_name)

    df = pd.DataFrame({'id': names, 'label': labels})
    parent = Path(images_paths[0]).parent.parent.parent
    path = os.path.join(os.path.expanduser('~'), parent, 'all_data.csv')
    print("Found %d different class files. If you have more than %d classes you should split your images as different "
          "files. " % (len(df.label.unique()), len(df.label.unique())))
    df.to_csv(path, index=False)
    return df