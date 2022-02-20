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
        self.classes = self.train_df.label.unique().tolist()


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
            counts = train_df.label.value_counts()
            count_dict = counts.to_dict()
            print("Class distribution : %s ,Balancing process is started." % (count_dict))
            with tf.device('/device:GPU:0'):
                with tf.device('/device:GPU:0'):
                    train_generator = BalancedDataGenerator(self.images_folder, train_df, train_datagen,
                                                            self.target_size,
                                                            self.batch_size, "categorical")
                    print("Training data has made balance.")
                    return train_generator, val_generator, test_generator,self.classes


class BalancedDataGenerator(Sequence):

    def __init__(self, data_path, dataframe, datagen, target_size, batch_size, class_mode='categorical'):
        self.data_path = data_path
        self.dataframe = dataframe
        self.datagen = datagen
        self.target_size = target_size
        self.class_mode = class_mode
        self.batch_size = batch_size
        X = self.dataframe.id
        y = self.dataframe.label
        X = np.array(X)
        y = np.array(y)

        self.batch_size = min(self.batch_size, X.shape[0])
        self.gen, self.steps_per_epoch = balanced_batch_generator(X.reshape(X.shape[0], -1), y,
                                                                  sampler=RandomOverSampler(),
                                                                  batch_size=self.batch_size, keep_sparse=True)
        self._shape = (self.steps_per_epoch * self.batch_size, *X.shape[1:])

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        x_batch, y_batch = self.gen.__next__()
        x_batch = x_batch.reshape(-1)
        df = pd.DataFrame()
        df['id'] = x_batch
        df['label'] = y_batch
        print("Training Dataset : ")
        return self.datagen.flow_from_dataframe(dataframe=df,
                                                directory=self.data_path,
                                                x_col='id',
                                                y_col='label',
                                                subset="training",
                                                shuffle=True,
                                                seed=42,
                                                target_size=(self.target_size, self.target_size),
                                                class_mode=self.class_mode,
                                                batch_size=self.batch_size).next()


