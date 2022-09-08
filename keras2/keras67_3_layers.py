from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

# model.trainable =False
# Total params: 17
# Trainable params: 0
# Non-trainable params: 17

# for layers in model.layers:
#     layers.trainable = False
# Total params: 17
# Trainable params: 0
# Non-trainable params: 17

model.layers[0].trainable=False     #Dense
model.layers[1].trainable=False     #Dense_1
model.layers[2].trainable=False     #Dense_2 
 
 
# Total params: 17
# Trainable params: 11
# Non-trainable params: 6

model.summary()
print(model.layers)


