import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import inspect


class Custom_Model(tf.keras.Model):
    def _init_(self,  model_name, image_size, weights='imagenet'):
        super(Custom_Model, self).__init__()
        self.input_shape = image_size
        self.weights = weights
        self.model_name = model_name
        self.model_dictionary = {m[0]: m[1] for m in inspect.getmembers(tf.keras.applications, inspect.isfunction)}
        self.model_dictionary.pop('NASNetLarge')
        self.base_model = self.model_dictionary[self.model_name](input_shape=self.input_shape,
                                                                 include_top=False,weights = self.weights)
        self.base_model.trainable = False
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs,training=False):
        print(self.base_model.summary())
        x = self.base_model(inputs,training=training)
        x = tf.keras.layers.GlobalAveragePooling2D(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = self.dense(x)
        return outputs

    def build_graph(self):
        x = tf.keras.layers.Input(shape=self.input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
