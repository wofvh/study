from flask import Flask, render_template, request
import os
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten , Dropout,MaxPooling2D,LSTM
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import accuracy_score
from flask import Flask, url_for

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'D:\study_data\_data/test/test'

#업로드 HTML 렌더링
@app.route('/')
def render_file():
   return render_template('start1.html')

#파일 업로드 처리
@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
        f = request.files['file']
        #저장할 경로 + 파일명
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        
        season = ImageDataGenerator(
        rescale=1./255)

        season1 = season.flow_from_directory(
        'D:\study_data\_data/test/',
        target_size=(150,150),# 크기들을 일정하게 맞춰준다.
        batch_size=4000,
        class_mode='categorical', 
        # color_mode='grayscale', #디폴트값은 컬러
        shuffle=True,
        )
        print(season1[0][0])

        np.save('d:/study_data/_save/_npy/personaltest_test.npy', arr=season1[0][0])


        #1. 데이터
        season = np.load('d:/study_data/_save/_npy/personaltest_test.npy')
        x_train = np.load('d:/study_data/_save/_npy/project_train7_x.npy')
        y_train = np.load('d:/study_data/_save/_npy/project_train7_y.npy')
        x_test = np.load('d:/study_data/_save/_npy/project_test7_x.npy')
        y_test = np.load('d:/study_data/_save/_npy/project_test7_y.npy')

        print(x_train.shape)            # (2000, 150, 150, 3)
        print(y_train.shape)            # (2000,)
        print(x_test.shape)             # (550, 150, 150, 3)
        print(y_test.shape)             # (550,)

        # x_train = x_train.reshape(2000,450,150)
        # x_test = x_test.reshape(550,450,150)


        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.layers import Dense, Conv2D, Flatten , Dropout,MaxPooling2D,LSTM


        #2. 모델 
        # model = Sequential()
        # model.add(Conv2D(64,(3,3), input_shape = (150,150,3), padding='same', activation='relu'))
        # model.add(MaxPooling2D(2,2))
        # model.add(Conv2D(128,(3,3), padding='same',activation='relu'))
        # model.add(MaxPooling2D(2,2))
        # model.add(Conv2D(128,(3,3), padding='same',activation='relu'))
        # model.add(Flatten())
        # model.add(Dropout(0.5))
        # model.add(Dense(100,activation='relu'))
        # model.add(Dense(100,activation='relu'))
        # model.add(Dense(7,activation='softmax'))
            
        # model.save("./_save/project_save_model.h1")
        
        # #3. 컴파일.훈련

        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])

        # earlystopping =EarlyStopping(monitor='loss', patience=15, mode='auto', 
        #             verbose=1, restore_best_weights = True)     

        # hist = model.fit(x_train,y_train, epochs=50,validation_split=0.3,verbose=2,batch_size=32,
        #                 callbacks=[earlystopping]) 
        model = load_model('C:\study\_save/project999.h5')
        
        #4. 예측
        # accuracy = model.history['accuracy']
        # val_accuracy = model.history['val_accuracy']
        # loss = model.history['loss']
        # val_loss = model.history['val_loss']

        # print('loss : ',loss[-1])
        # print('accuracy : ', accuracy[-1])

        # ############################################
        # loss = model.evaluate(x_test, y_test)
        y_predict = model.predict(season)

        y_test = np.argmax(y_test, axis= 1)
        y_predict = np.argmax(y_predict, axis=1) 
        print('predict : ',y_predict)
        
        
        if y_predict[0] == 0:
            wh_result='<우박>  내륙에는 우박이 떨어지는 곳이 있겠습니다. 각별히 유의하기 바랍니다.  '
        elif  y_predict[0] ==1 :
            wh_result='<번개>  풍과 천둥번개가 동반될 수 있습니다. 틈나는 대로 날씨 변화를 점검해주시기 바랍니다.'
        elif  y_predict[0] ==2 :
            wh_result='<비> 비구름대가 발달하면서 내륙에는 비가 오는 곳이 있겠습니다.반드시 우산을 챙기시기 바랍니다. '
        elif  y_predict[0] ==3 :
            wh_result='<무지개>  소나기가 지나간 하늘에 무지개가 떴습니다.'
        elif  y_predict[0] ==4 :
            wh_result='<맑은날> 고기압의 영향으로 대체로 날은 맑겠습니다.미세먼지 농도는 좋음 단계로 야외 활동하기 좋습니다. '        
        elif  y_predict[0] ==5 :
            wh_result='<황사> 이번 베이징의 황사는 중국의 황사경보 4단계 중 낮은 청색경보 수준이라, 한국에는 약한 수준의 황사정도가 예상됩니다.'        
        elif  y_predict[0] ==6 :
             wh_result='<눈>  찬 대륙고기압이 우리나라에 확장되면서 기온이 급격히 낮아지고 대설이 예상됩니다.'   

        # wh1 = y_predict[0]

        return render_template('end1.html', wh=wh_result,sa=app.config['UPLOAD_FOLDER'])

    
if __name__ == '__main__':
    #서버 실행
   app.run(debug = True)

