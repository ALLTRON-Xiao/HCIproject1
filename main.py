import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, MobileNetV3Large, VGG16
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

IMG_SIZE = 224

x_train_resized = tf.image.resize(np.expand_dims(x_train, -1), [IMG_SIZE, IMG_SIZE])
x_test_resized = tf.image.resize(np.expand_dims(x_test, -1), [IMG_SIZE, IMG_SIZE])

x_train_rgb = tf.image.grayscale_to_rgb(x_train_resized)
x_test_rgb = tf.image.grayscale_to_rgb(x_test_resized)

print(f"train data shape: {x_train_rgb.shape}")
print(f"test data shape: {x_test_rgb.shape}")

def build_finetune_model(base_model, img_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=10):

    # frozen all
    # base_model.trainable = False

    # part frozen for res and mob
    base_model.trainable = True
    for layer in base_model.layers[:-10]: # unfrozen 10 if ResNet, 20 if Mob
        layer.trainable = False

    # if VGG16 use this part frozen
    # base_model.trainable = True

    # # unfrozen block5 conv and pool
    # set_trainable = False
    # for layer in base_model.layers:
    #     if layer.name == 'block5_conv1':
    #         set_trainable = True
    #     if set_trainable:
    #         layer.trainable = True
    #     else:
    #         layer.trainable = False

    

    # add layers to retrain
    inputs = layers.Input(shape=img_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model
# 1: ResNet50
print("\nLoading ResNet50...")
resnet_base = ResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
finetune_model = build_finetune_model(resnet_base)

# 2: MobileNetV3 
# print("\nLoading MobileNetV3Large...")
# mobilenet_base = MobileNetV3Large(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
# finetune_model = build_finetune_model(mobilenet_base)

# 3: VGG16
# print("\nLoading VGG16...")
# vgg16_base = VGG16(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
# finetune_model = build_finetune_model(vgg16_base)

# if fullfrozen
# finetune_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
#                               loss='categorical_crossentropy',
#                               metrics=['accuracy'])

#if part frozen, lower learning rate
finetune_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])


finetune_model.summary()

# retrain
history_finetune = finetune_model.fit(x_train_rgb, y_train,
                                             epochs=5,
                                             batch_size=64,
                                             validation_data=(x_test_rgb, y_test))

# evlauate
print("\nEvaluating fine-tuned model on the test set...")
score_finetune = finetune_model.evaluate(x_test_rgb, y_test, verbose=0)
print(f"Fine-tuned model test loss: {score_finetune[0]}")
print(f"Fine-tuned model test accuracy: {score_finetune[1]}")