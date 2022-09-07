import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D,MaxPooling2D,Flatten
from keras.applications import VGG16
# model = VGG16(weights='imagenet',include_top=True,
#                 input_shape=(224,224,3))
#####################include_top =True#############################
# 1. FC layer 원래꺼 그대로 쓴다. 
# 2. input_shape = (224,224,3) 고정값, -바꿀 수 없다. 

#  input_1 (InputLayer)        [(None, 224, 224, 3)]     0

#  block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792

# ----------------------------------------------------------------
#  flatten (Flatten)           (None, 25088)             0

#  fc1 (Dense)                 (None, 4096)              102764544

#  fc2 (Dense)                 (None, 4096)              16781312

#  predictions (Dense)         (None, 1000)              4097000

# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
model = VGG16(weights='imagenet',include_top=False,
              input_shape=(32,32,3))


model_cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    MaxPooling2D(2, 2), 
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2), 
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2), 
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2), 
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2), 
    Flatten(), 
    Dense(512, activation='relu'), 
    Dense(32, activation='relu'), 
    Dense(1, 'sigmoid')
])
model.summary()
print(len(model.weights))
print(len(model.trainable_weights))

