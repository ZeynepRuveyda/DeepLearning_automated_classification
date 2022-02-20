import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import inspect


class Custom_Model(tf.keras.Model):
    def __init__(self, model_name,IMG_SHAPE):
        super(Custom_Model, self).__init__()
        # define all layers in init
        self.IMG_SHAPE = IMG_SHAPE
        self.model_name = model_name
        self.dense = tf.keras.layers.Dense(1, activation=tf.nn.softmax)
        self.flat = tf.keras.layers.Flatten(name="flatten")
        self.model_dictionary = {m[0]: m[1] for m in inspect.getmembers(tf.keras.applications, inspect.isfunction)}
        self.model_dictionary.pop('NASNetLarge')
        self.base_model = self.model_dictionary[self.model_name](input_shape=self.IMG_SHAPE, include_top=False,
                                                                 weights='imagenet')
        self.base_model.trainable = False
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, input_tensor,training=False):
        x = self.base_model(input_tensor)
        x = self.global_average_layer(x)
        if training:
            x = self.dropout(x, training= training)
            outputs = self.dense(x)
        model = tf.keras.Model(input_tensor, outputs)
        return model
