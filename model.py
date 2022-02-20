import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import inspect


class Custom_Model(tf.keras.Model):
    def __init__(self, model_name, IMG_SHAPE):
        super(Custom_Model, self).__init__()
        # define all layers in init
        self.model_name = model_name
        self.IMG_SHAPE = IMG_SHAPE
        self.dense = tf.keras.layers.Dense(1)
        self.flat = tf.keras.layers.Flatten(name="flatten")
        self.model_dictionary = {m[0]: m[1] for m in inspect.getmembers(tf.keras.applications, inspect.isfunction)}
        self.model_dictionary.pop('NASNetLarge')
        self.base_model = self.model_dictionary[self.model_name](input_shape=self.IMG_SHAPE, include_top=False,
                                                                 weights='imagenet')
        self.base_model.trainable = False

    def call(self, input_tensor):
        x = self.base_model(input_tensor, training=False)
        outputs = self.dense(x)
        return outputs

    def summary(self):
        x = tf.keras.layers.Input(shape=self.IMG_SHAPE)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()
