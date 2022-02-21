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
import seaborn as sns
import math
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import tensorflow as tf
import inspect
from model import *
# Creating csv files for all image with their nm_classes. Classes will be added as folder name of image .
# Getting all images into a one folder
# Return two data csv and all images data path.
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

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
                im = cv2.imread(os.path.join(os.path.expanduser('~'), f_path, img_name), -1)
                img_name = img_name.replace(".png", ".jpg")
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
            name = latest_file_name + '_' + Path(image_path).stem + '.jpg'
            names.append(name)
            labels.append(latest_file_name)

    df_all = pd.DataFrame({'id': names, 'label': labels})
    parent = Path(images_paths[0]).parent.parent.parent
    path_csv = os.path.join(os.path.expanduser('~'), parent, 'all_data.csv')
    print("Found %d different class folders. If you have more than %d nm_classes you should split your images as different"
          " folders in zip file." % (len(df_all.label.unique()), len(df_all.label.unique())))
    df_all.to_csv(path_csv, index=False)
    train_df, test_df = test_train_split(path_csv)
    return train_df, test_df, path_all_images


def test_train_split(df_dir):
    df = pd.read_csv(df_dir)
    train_df, test_df = train_test_split(df, test_size=0.1)
    parent = Path(df_dir).parent
    train_df_path = os.path.join(os.path.expanduser('~'), parent, 'TRAIN_DF.csv')
    test_df_path = os.path.join(os.path.expanduser('~'), parent, 'TEST_DF.csv')
    train_df.to_csv(train_df_path, index=False)
    test_df.to_csv(test_df_path, index=False)
    return train_df, test_df


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


def random_over_sampling(train_df, all_image_path):
    train_df.reset_index(drop=True, inplace=True)
    index = train_df.index
    counts = train_df.label.value_counts()
    count_dict = counts.to_dict()
    max_class = max(count_dict.values())
    max_key = max(count_dict, key=count_dict.get)
    count_dict.pop(max_key)
    names = []
    labels = []
    for key, value in count_dict.items():
        dif = max_class - value
        label_ind = list(index[train_df["label"] == key])
        if len(label_ind) < dif:
            inds = random.choices(label_ind, k=dif)
            j = 0
            for i in inds:
                sample = train_df.iloc[i]
                names.append(str(j) + '_copy_' + sample['id'])
                labels.append((sample['label']))
                im = cv2.imread(os.path.join(os.path.expanduser('~'), all_image_path, sample['id']))
                cv2.imwrite(os.path.join(os.path.expanduser('~'), all_image_path, str(j) + '_copy_' + sample['id']), im)
                j = j + 1
        else:
            inds = random.sample(label_ind, dif)

            for i in inds:
                sample = train_df.iloc[i]
                names.append('copy_' + sample['id'])
                labels.append((sample['label']))
                im = cv2.imread(os.path.join(os.path.expanduser('~'), all_image_path, sample['id']), -1)
                cv2.imwrite(os.path.join(os.path.expanduser('~'), all_image_path, 'copy_' + sample['id']), im)

    new_names = list(train_df.id) + names
    new_labels = list(train_df.label) + labels
    new_train_df = pd.DataFrame({'id': new_names, 'label': new_labels})
    parent = Path(all_image_path).parent
    path_csv = os.path.join(os.path.expanduser('~'), parent, 'TRAIN_DF.csv')
    new_train_df.to_csv(path_csv, index=False)

    return new_train_df


def create_class_weight(labels_dict, n_classes):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()
    x = 0
    for key in keys:
        y = labels_dict[key]
        score = (total / n_classes) * (1/y)
        class_weight[x] = score
        x += 1
    return class_weight

def model_selection(train_generator,validation_generator,im_size,nm_classes):
    model_dictionary = {m[0]: m[1] for m in inspect.getmembers(tf.keras.applications, inspect.isfunction)}
    model_dictionary.pop('NASNetLarge')
    model_benchmarks = {'model_name': [], 'num_model_params': [] ,'validation_accuracy': []}
    for model_name, model in tqdm(model_dictionary.items()):

        model_ = Custom_Model(model_name,im_size,nm_classes)
        # custom modifications on top of pre-trained model
        model_ = model_.forward()
        base_learning_rate = 0.0001
        model_.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        history = model_.fit(train_generator, epochs=3, validation_data=validation_generator,)

        model_benchmarks['model_name'].append(model_name)
        model_benchmarks['num_model_params'].append(model_.count_params())
        model_benchmarks['validation_accuracy'].append(history.history['val_accuracy'][-1])
        benchmark_df = pd.DataFrame(model_benchmarks)
        benchmark_df.to_csv('benchmark_df.csv', index=False)  # write results to csv file

    return benchmark_df

def benchmark_visualization(benchmark_df):
    # Loop over each row and plot the num_model_params vs validation_accuracy
    markers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X",
               "D", "d", "|", "_", 4, 5, 6, 7, 8, 9, 10, 11]
    plt.figure(figsize=(7, 5))
    for row in benchmark_df.itertuples():
        plt.scatter(row.num_model_params, row.validation_accuracy, label=row.model_name, marker=markers[row.Index],
                    s=150, linewidths=2)
    plt.xscale('log')
    plt.xlabel('Number of Parameters in Model')
    plt.ylabel('Validation Accuracy after 3 Epochs')
    plt.title('Accuracy vs Model Size')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left');  # Move legend out of the plot

    return plt.show()


def metrics_visualization(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')

    return plt.show()


def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
  print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
  print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
  print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
  print('Total Fraudulent Transactions: ', np.sum(cm[1]))

  print("_______________________________________________________________________")
  print(classification_report(labels, predictions))

  return plt.show