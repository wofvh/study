# from click import argument
# from sklearn.utils import shuffle
from colorsys import yiq_to_rgb
from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

train_dataen = ImageDataGenerator(               
    rescale=1./255,                             
                                                
    horizontal_flip=True,                       
    # vertical_flip=True,                         
    width_shift_range=0.1,                      
    height_shift_range=-0.1,                       
    rotation_range=5,                          
    zoom_range=0.1,                            
    # shear_range=0.7,                            
    fill_mode='nearest'                         
)

augument_size = 20                   # 반복횟수
randidx =np.random.randint(x_train.shape[0],size=augument_size)

print(x_train.shape[0])         # 60000
print(y_train.shape[0])         # 60000
print(x_test.shape[0])          # 10000
print(y_test.shape[0])          # 10000
print(randidx.shape)            # 40000
print(randidx)                  # [39683 20510 12895 ... 24908 55852  1491] 
print(np.min(randidx),np.max(randidx))      # random 함수 적용가능. 
print(type(randidx))            # <class 'numpy.ndarray'> 기본적으로 리스트 형태.       


x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy()

print(x_augumented.shape)       # (40000, 28, 28)
print(y_augumented.shape)       # (40000,)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

x_augumented = x_augumented.reshape(x_augumented.shape[0],
                                    x_augumented.shape[1],
                                    x_augumented.shape[2],
                                    1)
import time
start_time = time.time()
print('시작')

x_augumented = train_dataen.flow(x_augumented,y_augumented,
                                batch_size = augument_size,
                                save_to_dir= 'D:\study_data/_temp/',
                                shuffle=False).next()[0]

end_time = time.time() -start_time

print(end_time,"걸린시간 ",round(end_time,3),"초")

# print(x_augumented)
# print(x_augumented.shape)

# x_train =np.concatenate((x_train,x_augumented))
# y_train =np.concatenate((y_train,y_augumented))

# print(x_train.shape,y_train.shape)


# print(x_train[0].shape)                         #(28, 28)
# print(x_train[0].reshape(28*28).shape)          # (784,)
# print(np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1,28,28,1).shape)          # (100, 28, 28, 1)
# # reshape  # (100, 28, 28, 1) (열, reshape,reshape,reshape)


# print(np.zeros(augument_size))
# print(np.zeros(augument_size).shape)
# print(np.tile(x_train[0].reshape(28*28), augument_size).shape)                      # (31360000,)
# print(np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1,28,28,1).shape)  # (40000, 28, 28, 1)


# x_data = train_dataen.flow(
#     np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1,28,28,1),   # x
#     np.zeros(augument_size),                                                 # y
#     batch_size=augument_size,
#     shuffle=True).next()   # < 알아보기 
# ##############################next사용 ###################################
# # print(x_data)
# # print(x_data[0])
# # print(x_data[0].shape)               # (100, 28, 28, 1)
# # print(x_data[1].shape)               # (100,)
# ##############################next 미사용 #####################################
# print(x_data)
# print(x_data[0])
# print(x_data[0][0].shape)               # (100, 28, 28, 1)
# print(x_data[0][1].shape)               # (100,)



# import matplotlib.pyplot as plt
# plt.figure(figsize=(25,25))
# for i in range(20) :
#     plt.subplot(2,10,i+1)
#     plt.axis('off')
#     # plt.imshow(x_data[0][i], cmap='gray')        # next사용
#     plt.imshow(x_data[0][i], cmap='gray')       # next미사용
    
# plt.show()
