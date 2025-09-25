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
    """
    构建一个用于微调的通用模型。
    Args:
        base_model: 预加载的、不含顶层的Keras模型 (例如 ResNet50)
        img_shape: 输入图像的尺寸
        num_classes: 输出类别的数量
    """
    # frozen all
    base_model.trainable = False

    # part frozen

    # base_model.trainable = True
    # for layer in base_model.layers[:-20]: # 冻结除了最后10层之外的所有层
    #     layer.trainable = False

    # add layers to retrain
    inputs = layers.Input(shape=img_shape)
    x = base_model(inputs, training=False) # 在推理模式下运行base_model
    x = layers.GlobalAveragePooling2D()(x) # 添加全局平均池化层
    x = layers.Dropout(0.3)(x) # 添加Dropout以防止过拟合
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x) # 新的输出层

    model = models.Model(inputs, outputs)
    return model
# 模型1: ResNet50
# print("\nLoading ResNet50...")
# resnet_base = ResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
# finetune_model_resnet = build_finetune_model(resnet_base)

# 模型2: MobileNetV3 
# print("\nLoading MobileNetV3Large...")
# mobilenet_base = MobileNetV3Large(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
# finetune_model_mobilenet = build_finetune_model(mobilenet_base)

#3 VGG
vgg16_base = VGG16(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
finetune_model_vgg16 = build_finetune_model(vgg16_base)

# 编译微调模型
finetune_model_vgg16.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])

finetune_model_vgg16.summary()

# 训练微调模型
# 注意：由于图像尺寸变大且模型更复杂，训练会比自定义CNN慢得多
history_finetune = finetune_model_vgg16.fit(x_train_rgb, y_train,
                                             epochs=5, # 先用较少的epoch进行尝试
                                             batch_size=64,
                                             validation_data=(x_test_rgb, y_test))

# 评估微调模型
print("\nEvaluating fine-tuned model on the test set...")
score_finetune = finetune_model_vgg16.evaluate(x_test_rgb, y_test, verbose=0)
print(f"Fine-tuned model test loss: {score_finetune[0]}")
print(f"Fine-tuned model test accuracy: {score_finetune[1]}")