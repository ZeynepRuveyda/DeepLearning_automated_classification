import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import inspect


class Custom_Model():
    def __init__(self, model_name,im_size,nm_classes):

        # define all layers in init
        self.IMG_SHAPE = im_size + (3,)
        self.model_name = model_name
        self.nm_classes = nm_classes
        self.dense = tf.keras.layers.Dense(self.nm_classes, activation = 'sigmoid')
        self.flat = tf.keras.layers.Flatten(name="flatten")
        self.model_dictionary = {m[0]: m[1] for m in inspect.getmembers(tf.keras.applications, inspect.isfunction)}
        self.base_model = self.model_dictionary[self.model_name](input_shape=self.IMG_SHAPE, include_top=False,
                                                                 weights='imagenet')
        self.base_model.trainable = False
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(0.2)

    def forward(self):
        input_ = tf.keras.Input(shape=self.IMG_SHAPE)
        x = self.base_model(input_,training=False)
        x = self.global_average_layer(x)
        #x = self.dropout(x)
        outputs = self.dense(x)
        model = tf.keras.Model(input_,outputs)
        return model
