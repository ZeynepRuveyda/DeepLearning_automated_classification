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

    def __init__(self, images_folder, test_df, train_df, batch_size, target_size):
        self.images_folder = images_folder
        self.test_df = test_df
        self.train_df = train_df
        self.batch_size = batch_size
        self.target_size = target_size
        self.classes = len(self.train_df.label.unique().tolist())

    def data_generator(self):
        with tf.device('/device:GPU:0'):
            train_df, eval_df = train_test_split(self.train_df, test_size=0.2)
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
                directory=self.images_folder,
                x_col="id",
                y_col=None,
                target_size=(self.target_size, self.target_size),
                batch_size=self.batch_size,
                shuffle=False,
                class_mode=None
            )
            print("Validation Dataset : ")
            val_generator = val_datagen.flow_from_dataframe(

                dataframe=eval_df,
                directory=self.images_folder,
                x_col="id",
                y_col="label",
                target_size=(self.target_size, self.target_size),
                batch_size=self.batch_size,
                shuffle=True,
                seed=42,
                class_mode="categorical"
            )
        counts = train_df.label.value_counts()
        count_dict = counts.to_dict()
        class_weights = create_class_weight(count_dict,self.classes)

        if balance_check(self.train_df):

            with tf.device('/device:GPU:0'):
                print("Training Dataset : ")
                train_generator = train_datagen.flow_from_dataframe(
                    dataframe=train_df,
                    directory=self.images_folder,
                    x_col="id",
                    y_col="label",
                    target_size=(self.target_size, self.target_size),
                    batch_size=self.batch_size,
                    shuffle=True,
                    seed=42,
                    class_mode="categorical"
                )
                return train_generator, val_generator, test_generator,self.classes
        else:


            print("Class distribution : %s ,Balancing process is started." % (count_dict))
            train_df_new = random_over_sampling(train_df,self.images_folder)
            with tf.device('/device:GPU:0'):
                print("Training Dataset : ")
                train_generator = train_datagen.flow_from_dataframe(
                    dataframe=train_df_new,
                    directory=self.images_folder,
                    x_col="id",
                    y_col="label",
                    target_size=(self.target_size, self.target_size),
                    batch_size=self.batch_size,
                    shuffle=True,
                    seed=42,
                    class_mode="categorical"
                )
                counts = train_df_new.label.value_counts()
                count_dict = counts.to_dict()

                print("Class distribution : %s ,Data made balance." % (count_dict))
                return train_generator, val_generator, test_generator, self.classes,class_weights





