import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from utils import *
import os
import pandas as pd
import numpy as np
from tensorflow.keras.utils import Sequence
from imblearn.over_sampling import RandomOverSampler
from imblearn.tensorflow import balanced_batch_generator
from sklearn.model_selection import train_test_split


class CustomDataGenerator:

    def __init__(self, images_folder_dir, test_df, train_df, eval_df, batch_size, target_size):
        self.images_folder_dir = images_folder_dir
        self.test_df = test_df
        self.train_df = train_df
        self.eval_df = eval_df
        self.batch_size = batch_size
        self.target_size = target_size

    def data_generator(self):
        with tf.device('/device:GPU:0'):
            train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            val_datagen = ImageDataGenerator(
                rescale=1. / 255
            )
            print("Test Dataset : ")
            test_generator = val_datagen.flow_from_dataframe(
                dataframe= self.test_df,
                directory=self.images_folder_dir,
                x_col="id",
                y_col="label",
                target_size=self.target_size,
                batch_size=self.batch_size,
                shuffle=False,
                class_mode=None
            )
            print("Validation Dataset : ")
            val_generator = val_datagen.flow_from_dataframe(

                dataframe=self.eval_df,
                directory=self.images_folder_dir,
                x_col="id",
                y_col="label",
                target_size=self.target_size,
                batch_size=self.batch_size,
                shuffle=True,
                seed=42,
                class_mode="categorical"
            )
            print("Training Dataset : ")
            train_generator = train_datagen.flow_from_dataframe(
                dataframe=self.train_df,
                directory=self.images_folder_dir,
                x_col="id",
                y_col="label",
                target_size=self.target_size,
                batch_size=self.batch_size,
                shuffle=True,
                seed=42,
                class_mode="categorical"
            )
            return train_generator, val_generator, test_generator





