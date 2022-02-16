import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import inspect


class ModelSubClassing(tf.keras.Model):
    def __init__(self,model_name,IMG_SHAPE):
        super(ModelSubClassing, self).__init__()
        # define all layers in init
        self.model_name = model_name
        self.IMG_SHAPE = IMG_SHAPE
        self.dense = tf.keras.layers.Dense(1)

    def call(self, input_tensor):

        model_dictionary = {m[0]: m[1] for m in inspect.getmembers(tf.keras.applications, inspect.isfunction)}
        model_dictionary.pop('NASNetLarge')
        base_model = model_dictionary[self.model_name](input_shape=self.self.IMG_SHAPE,include_top=False,weights = 'imagenet' )
        base_model.trainable = False
        print(base_model.summary())
        x = base_model(input_tensor,training=False)
        x = tf.keras.layers.GlobalAveragePooling2D(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = self.dense(x)
        return outputs

    def summary(self):
        x = tf.keras.layers.Input(shape=self.self.IMG_SHAPE)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()
