import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

class Custom_Model(tf.keras.Model):
    def __init__(self,input_shape,backbone,weights = 'imagenet'):
        self.input_shape = input_shape
        self.weights = weights
        self.backbone = backbone


    def call(self, inputs):


