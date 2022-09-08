from keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from keras.applications.resnet import preprocess_input, decode_predictions # preprocess_input : 이미지 전처리, decode_predictions : 예측 결과를 해석
import numpy as np

model = ResNet50(weights='imagenet') # default : imagenet, 

img_path = 'D:\study_data\_data\dog/sheep_dog.png'

img = image.load_img(img_path, target_size=(224, 224)) # 이미지를 불러오고 크기를 변경한다.
print(type(img)) # <class 'PIL.Image.Image'>

x = image.img_to_array(img) # 이미지를 배열로 변환한다.
print('===============image.img_to_array(img)================')
print(x, '/n', x.shape)
print(np.max(x), np.min(x)) # 255.0 13.0


x = np.expand_dims(x, axis=0) # 차원을 추가한다.
print('===============image.img_to_array(img)================')
print(x, '/n', x.shape)


x = preprocess_input(x) # 이미지 전처리
print('===============image.img_to_array(img)================')
print(x, '/n', x.shape)
print(np.max(x), np.min(x)) # 151.061 -108.68

preds = model.predict(x)
print(preds,'\n', preds.shape) # (1, 1000)

print('Predicted:', decode_predictions(preds, top=3)[0]) # decode_predictions : 예측 결과를 해석