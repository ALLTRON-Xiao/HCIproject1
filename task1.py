import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2, l1
import numpy as np
import matplotlib.pyplot as plt

(x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = keras.datasets.mnist.load_data()

# 归一化
x_train = x_train_orig.astype("float32") / 255.0
x_test = x_test_orig.astype("float32") / 255.0

# one-hot
y_train = keras.utils.to_categorical(y_train_orig, num_classes=10)
y_test = keras.utils.to_categorical(y_test_orig, num_classes=10)

def cnn_dropout():
    model = models.Sequential()

    model.add(layers.Input(shape=(28, 28, 1)))

    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    # Dropout
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation="relu"))

    model.add(layers.Dense(10, activation="softmax"))

    return model

def cnn_l1():
    model = models.Sequential()

    model.add(layers.Input(shape=(28, 28, 1)))

    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    # L1 and penalty
    model.add(layers.Dense(128, activation="relu", kernel_regularizer=l1(0.01)))

    model.add(layers.Dense(10, activation="softmax"))

    return model

def cnn_l2():
    model = models.Sequential()

    model.add(layers.Input(shape=(28, 28, 1)))

    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    # L2 and penalty
    model.add(layers.Dense(128, activation="relu", kernel_regularizer=l2(0.01)))

    model.add(layers.Dense(10, activation="softmax"))

    return model

x_train_cnn = np.expand_dims(x_train, -1)
x_test_cnn = np.expand_dims(x_test, -1)

# need to choose
custom_model = cnn_dropout()
# custom_model = cnn_l1()
# custom_model = cnn_l2()
custom_model.summary()

# optimizers and learning rate
optimizer1 = keras.optimizers.Adam(learning_rate=0.001)
optimizer2 = keras.optimizers.SGD(learning_rate=0.01)

# need to choose
print("\ndropout op1 Model")
print("\nCompiling model with Adam optimizer...")
custom_model.compile(loss="categorical_crossentropy", optimizer=optimizer1, metrics=["accuracy"])
# custom_model.compile(loss="categorical_crossentropy", optimizer=optimizer2, metrics=["accuracy"])

history = custom_model.fit(x_train_cnn, y_train, batch_size=128, epochs=10, validation_split=0.1)
custom_model.save('cnn_dropout_adam.keras')
# evaluate on test
print("\nEvaluating model on the test set...")
score = custom_model.evaluate(x_test_cnn, y_test, verbose=0)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")