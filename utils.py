# importing required libraries
import os
import pandas as pd
from pathlib import Path
from zipfile import ZipFile
import shutil
import cv2
from sklearn.model_selection import train_test_split
import time
import random
import math
import numpy as np
# Creating csv files for all image with their classes. Classes will be added as folder name of image .
# Getting all images into a one folder
# Return two data csv and all images data path.

def create_dataset(dir_zip):
    image_types = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    with ZipFile(dir_zip, 'r') as zipObj:
        # Get list of files names in zip
        folder_paths = zipObj.namelist()
        parent = Path(dir_zip).parent
        path_all_images = os.path.join(os.path.expanduser('~'), parent, 'ALL_IMAGES')
        zipObj.extractall(path_all_images)

    img_folders = next(os.walk(path_all_images))[1]

    for f in img_folders:
        f_path = os.path.join(os.path.expanduser('~'), path_all_images, f)
        files = os.listdir(f_path)
        for img_name in files:
            extension = os.path.splitext(img_name)[1]
            if extension in image_types:
                img_name = img_name.replace(".png",".jpg")
                im = cv2.imread(os.path.join(os.path.expanduser('~'), f_path, img_name), -1)
                im_name = f + "_" + img_name
                cv2.imwrite(os.path.join(os.path.expanduser('~'), path_all_images, im_name), im)
        shutil.rmtree(f_path)

    images_paths = []
    for f in folder_paths:
        path_csv = os.path.join(os.path.expanduser('~'), dir_zip, f)
        images_paths.append(path_csv)

    names = []
    labels = []

    for image_path in images_paths:
        filename, file_extension = os.path.splitext(image_path)
        latest_file = Path(image_path).parent.absolute()
        latest_file_name = Path(latest_file).stem

        if file_extension in image_types:
            name = latest_file_name + '_' + Path(image_path).stem + file_extension
            names.append(name)
            labels.append(latest_file_name)

    df_all = pd.DataFrame({'id': names, 'label': labels})
    parent = Path(images_paths[0]).parent.parent.parent
    path_csv = os.path.join(os.path.expanduser('~'), parent, 'all_data.csv')
    print("Found %d different class folders. If you have more than %d classes you should split your images as different"
          " folders in zip file." % (len(df_all.label.unique()), len(df_all.label.unique())))
    df_all.to_csv(path_csv, index=False)
    train_df,test_df = test_train_split(path_csv)
    return train_df, test_df, path_all_images


def test_train_split(df_dir):
    df = pd.read_csv(df_dir)
    train_df, test_df = train_test_split(df, test_size=0.1)
    parent = Path(df_dir).parent
    train_df_path = os.path.join(os.path.expanduser('~'), parent, 'TRAIN_DF.csv')
    test_df_path = os.path.join(os.path.expanduser('~'), parent, 'TEST_DF.csv')
    train_df.to_csv(train_df_path, index=False)
    test_df.to_csv(test_df_path, index=False)
    return train_df,test_df

def balance_check(train_df):
    counts = train_df.label.value_counts()
    count_dict = counts.to_dict()
    list_count = list(count_dict.values())
    if all(x == list_count[0] for x in list_count) == False:
        print("Training data is not balance.")
        time.sleep(3.0)
        return False
    else:
        print("Training data is balance.")
        return True

def random_over_sampling(train_df,all_image_path):
    train_df.reset_index(drop=True, inplace=True)
    index = train_df.index
    counts = train_df.label.value_counts()
    count_dict = counts.to_dict()
    max_class = max(count_dict.values())
    max_key = max(count_dict, key=count_dict.get)
    count_dict.pop(max_key)
    names = []
    labels = []
    for key,value in count_dict.items():
        dif = max_class - value
        label_ind = list(index[train_df["label"] == key])
        if len(label_ind) < dif:
            inds = random.choices(label_ind,k=dif)
            j = 0
            for i in inds:
                sample = train_df.iloc[i]
                names.append(str(j)+'_copy_' + sample['id'])
                labels.append((sample['label']))
                im = cv2.imread(os.path.join(os.path.expanduser('~'), all_image_path, sample['id']),-1)
                cv2.imwrite(os.path.join(os.path.expanduser('~'), all_image_path, 'copy_' + sample['id']+'.jpg'), im)
                j = j+1
        else:
            inds = random.sample(label_ind, dif)

            for i in inds:
                sample = train_df.iloc[i]
                names.append('copy_' + sample['id'])
                labels.append((sample['label']))
                im = cv2.imread(os.path.join(os.path.expanduser('~'), all_image_path, sample['id']), -1)
                cv2.imwrite(os.path.join(os.path.expanduser('~'), all_image_path, 'copy_' + sample['id']+'.jpg'), im)

    new_names = [x + y for x, y in zip(list(train_df.id), names)]
    new_labels = [x + y for x, y in zip(list(train_df.label), labels)]
    new_train_df = pd.DataFrame({'id': new_names, 'label': new_labels})


    return new_train_df

def create_class_weight(labels_dict, mu=0.15):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu * total / float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight






