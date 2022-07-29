import scipy.io
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib
import joblib
import pandas as pd
from flask import Flask, render_template, request
from werkzeug import secure_filename
import os
app = Flask(__name__)



# Google 주소 숫자 인식 모델 생성

# 로드 mat 파일
app.config['UPLOAD_FOLDER'] = 'D:\study_data\_data\season\dataset'
# 
#업로드 HTML 렌더링
@app.route('/')
def render_file():
   return render_template('upload.html')

#파일 업로드 처리
@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      #저장할 경로 + 파일명
      f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
      return 'uploads 디렉토리 -> 파일 업로드 성공!'

if __name__ == '__main__':
    #서버 실행
   app.run(debug = True)



# # 학습 데이터, 훈련 데이터
# X = train_data['X']
# y = train_data['y']

# # 매트릭스 1D 변환
# X = X.reshape(X.shape[0] * X.shape[1] * X.shape[2], X.shape[3]).T
# y = y.reshape(y.shape[0], )

# # 셔플(섞기)
# X, y = shuffle(X, y, random_state=42)

# # 학습 훈련 데이터 분리
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)



# import scipy.io
# from sklearn.utils import shuffle
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# # from sklearn.externals import joblib
# import joblib

# # 랜덤 포레스트 객체 생성 및 학습
# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)

# # 모델 저장
# joblib.dump(clf, './model/model.pk2')
