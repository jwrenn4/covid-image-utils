import tensorflow as tf
import numpy as np

def train_model(train_x, train_y):
    input_layer = tf.keras.layers.Input(train_x.shape[1:])
    x = tf.keras.layers.Conv2D(16, 3)(input_layer)
    x = tf.keras.layers.Conv2D(16, 3)(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(16, 3)(x)
    x = tf.keras.layers.Flatten()(x)
    output_layer = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
    model = tf.keras.models.Model(inputs = [input_layer], outputs = [output_layer])
    model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'], optimizer = 'adam')
    print(model.summary())
    model.fit(train_x, train_y, batch_size = 8, epochs = 10)
    return model