from keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from keras.applications.regnet import preprocess_input,decode_predictions
import numpy as np


model = ResNet50(weights='imagenet')

img_path = 'D:\study_data\_data\dog/나무늘보.png'
img = image.load_img(img_path,target_size=(224,224))

print(img)  # <PIL.Image.Image image mode=RGB size=224x224 at 0x22211C93A00>

x= image.img_to_array(img)      
print('===============image.img_to_array(img)================')
print(x,'\n',x.shape)       # (224, 224, 3)

x= np.expand_dims(x,axis=0)
print('===============np.expand_dims(x,axis=2)================')
print(x,'\n',x.shape)       # (1, 224, 224, 3)

x= preprocess_input(x)
print('===============preprocess_input(x)================')
print(x,'\n',x.shape)       # (1, 224, 224, 3)
print(np.min(x),np.max(x))

print('===============predict================')
preds =model.predict(x)
print(preds,'\n', preds.shape)

print('결과',decode_predictions(preds,top=5)[0])
# import matplotlib.pyplot as plt
# plt.imshow(x[0], cmap='gray')
# plt.show() 

