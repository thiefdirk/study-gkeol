import tensorflow as tf
import numpy as np
import os, sys
import time
from tensorflow.python.keras.layers import Dense, Embedding, LSTM, Concatenate, Dropout, Flatten
from tensorflow.python.keras.models import Input, Model, Sequential
from tensorflow.python.keras import optimizers
import os
from keras_self_attention import SeqSelfAttention
from attention import Attention


class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = Dense(units)
    self.W2 = Dense(units)
    self.V = Dense(1)

  def call(self, values, query): # 단, key와 value는 같음
    # query shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # score 계산을 위해 뒤에서 할 덧셈을 위해서 차원을 변경해줍니다.
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


# filename = "D:/study_data/_data/project/quote_generator/input_nolabel/input.txt"
filename = "D:\study_data\_data\project\quote_generator/input_label/input.txt"

# 문서 파일을 읽는다.
text = open(filename, 'rb').read().decode(encoding='utf-8')
#text  = text[:100000]

# 텍스트의 길이는 그 안에 있는 문자의 수다.
print("텍스트의 길이: {}자".format(len(text)))

# 처음 250자를 살펴본다.
print(text[:250])

# 파일의 고유 문자수를 출력.
vocab = sorted(set(text))
print(vocab)
print(len(vocab))

# 텍스트를 벡터화. char2idx, idx2char
char2idx = {u:i for i, u in enumerate(vocab)} # 0, 'A' -> {'\n': 0, ' ': 1, '!': 2, '$': 3, '&':  4....}
print(char2idx)

idx2char = np.array(vocab)
print(idx2char)
print(idx2char[:5])
print(text[:10])
tmp = [char2idx[c] for c in text]
print(tmp[:10])
# print(char2idx['F'], char2idx['i'],char2idx['r'], char2idx['s'], char2idx['t'])
text_as_int = np.array(tmp)
print(text_as_int[:100])

print ('{} ---- 문자들이 다음의 정수로 매핑되었습니다 ----> {}'.format(repr(text[:13]), text_as_int[:13]))

# 훈련 샘플과 타깃 만들기
# 단일 입력에 대해 원하는 문장의 최대 길이
seq_length = 80
examples_per_epoch = len(text) # seq_length
print(examples_per_epoch)

# 훈련 샘플/타깃 만들기
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
for num in char_dataset.take(5):
  print(num)
  print(idx2char[num])

# batch 메서드를 사용하면, 개별문자들을 원하는 크기의 시퀀스로 쉽게 변환할 수 있다.
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
for item in sequences.take(5):
  print(item)
  print(repr(''.join(idx2char[item])))
  
def split_input_target(chunk):
  input_text = chunk[:-1]
  target_text = chunk[1:]

  return input_text, target_text

dataset = sequences.map(split_input_target)

# 첫번째 샘플의 타깃 값을 출력한다.
for input_example, target_example in dataset.take(1):
  print('입력 데이터: ', repr(''.join(idx2char[input_example])))
  print('타깃 데이터: ', repr(''.join(idx2char[target_example])))
  print(input_example[:5])
  print(target_example[:5])

for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
  print("{:4d}단계".format(i))
  print(" 입력: {} {:s}".format(input_idx, repr(idx2char[input_idx])))
  print(" 예상 출력: {} {:s}".format(target_idx, repr(idx2char[target_idx])))

# 훈련 배치 생성
# 배치 크기
BATCH_SIZE = 64
# 데이터셋을 섞을 버퍼 크기
buffer_size = 10000

dataset = dataset.shuffle(buffer_size).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)

'''모델 설계
모델 정의: tf.keras.Sequential
- tf.keras.layers.Embedding: 입력층. embedding_dim 차원 벡터에 각 문자의 정수 코드를 매핑하는 훈련 가능한 검색 테이블
- tf.keras.layers.GRU: 크기가 units =rnn_units인 RNN타입. 여기서 LSTM을 사용할 수도 있음.
- tf.keras.layers.Dense: 크기가 vocab_size인 출력을 생성하는 출력층.'''

# 문자로 된 어휘 사전의 크기
vocab_size = len(vocab)
print(vocab_size)
embedding_dim = vocab_size
rnn_units = 1024

# def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
#   model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
#     tf.keras.layers.LSTM(rnn_units, return_sequences=True,
#                          stateful=True, kernel_initializer='glorot_uniform'),  # Xavier 정규분포 초기값 설정기
#     SeqSelfAttention(attention_activation='sigmoid'),
#     # Flatten(),
#     # Attention(64,use_scale=True),
#     tf.keras.layers.Dense(vocab_size)
#   ])
#   return model

epochs=100

#2. 모델구성
model = Sequential()
model.add(Embedding(len(vocab), embedding_dim, batch_input_shape=[BATCH_SIZE, None]))
model.add(LSTM(rnn_units, return_sequences=True,stateful=True, kernel_initializer='glorot_uniform'))
model.add(Attention(64))
model.add(Dense(len(vocab)))


#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

checkpoint_dir = 'D:\study_data\_save/_ModelCheckPoint/char_rnn_project/label/attention'


checkpoint_filename = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filename, save_weights_only=True)

history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])


# 체크포인트를 사용하여 훈련 중 체크포인트가 저장되도록 합니다.
# 체크포인트가 저장될 폴더
# 체크 포인트 파일이름



# 훈련실행
keep_training = 0 # 1: 모델을 불러와서 훈련을 계속할 때 0: 훈련을 처음할 때

# 문장 생성 모드
do_generate = 0  # 1: 문장 생성을 할 때, 0: 문장 생성 안 할 때

# 문장 생성을 할 때는 모델을 불러오지 않는다. 훈련을 하지도, 따라서 훈련 결과를 저장하지도 않는다.
# 지금까지 실행된 코드 결과(만들어진 변수값)를 기반으로, 
# 모델을 만들고(배치사이즈를 반드시 1로 해야 함, 안 그러면 에러 남), 체크포인트를 불러와서 훈련 weights를 적재한다.
# 모델을 그냥 불러와서 문장 생성을 하지 못하는 이유는, 훈련 때의 LSTM 인풋 데이터 shape과 문장 생성 시의 인풋 데이터
# shape이 다르기 때문이다. 문장생성 때는 배치를 하지 않고 하나씩 문자를 넣어줘서 예측을 하기 때문이다.

if do_generate == 0:
  if keep_training:
    # If you load model only for prediction (without training), you need to set compile flag to False:
    # model = load_model('saved_model/my_model', compile=False)
    model = tf.keras.models.load_model('D:\study_data\_save/_ModelCheckPoint/char_rnn_project/label/attention', custom_objects={'loss':loss})

  history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])    
  model.save('D:\study_data\_save/_ModelCheckPoint/char_rnn_project/label/attention')

# 텍스트 생성
# 최근 체크포인트 복원

# 문장 생성을 한다면, 모델을 별도로 만들어야 한다. (배치사이즈를 1로 해야 하기 때문에)
# 이때는 위의 온갓 char2idx, idx2char 등 온갖 변수값들이 필요하기 때문에, 문장생성 코드를 별도 파일로 작성하기가 난감하다.
# 그래서 조건문으로 처리.
if do_generate:
  print("latest_checkpoint: ", tf.train.latest_checkpoint(checkpoint_dir))

  model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
  #model.load_weights("./training_checkpoints\ckpt_100")

  model.build(tf.TensorShape([1, None]))
  #model.summary()

def generate_text(model, start_string):
  # 평가 단계 (학습된 모델을 사용하여 텍스트 생성)
  # 생성할 문자의 수
  num_generate = 200

  # 시작 문자열을 숫자로 변환(벡터화)
  input_eval =  [char2idx[s] for s in start_string]        # ex. ROMEO
  input_eval = tf.expand_dims(input_eval, 0)

  # 결과를 저장할 빈 문자열
  text_generated = []

  # 온도가 낮으면 더 예측 가능한 텍스트가 됩니다.
  # 온도가 높으면 더 의외의 텍스트가 됩니다.
  # 최적의 세팅을 찾기 위한 실험
  temperature = 1.5

  # 여기에서 배치 크기 == 1
  model.reset_states()

  for i in range(num_generate): # 0~ 1000
    predictions = model(input_eval)
    # 배치 차원 제거
    predictions = tf.squeeze(predictions, 0)

    # 범주형 분포를 사용하여 모델에서 리턴한 단어 예측
    predictions = predictions / temperature  # 
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy() # should check

    # 예측된 단어를 다음 입력으로 모델에 전달
    # 이전 은닉 상태와 함께
    input_eval = tf.expand_dims([predicted_id], 0)

    text_generated.append(idx2char[predicted_id])
  
  return (start_string + ''.join(text_generated))

# mytext = "ROMEO"
# myinput = [char2idx[s] for s in mytext] 
# print(myinput)
# # 차원을 하나 줄임: 2차원 배열 -> 1차원 배열
# myinput = tf.expand_dims(myinput, 0)
# print(myinput)
# print(myinput[0])
# print(tf.squeeze(myinput))


if do_generate:
  article = generate_text(model, start_string=u"가족 ")
  print(article)


# label epochs 700, temperture 1.4, 200자 생성, 가족, self-attention
# 스칼라 손실:      7.065503
# 가족 필필필필필필필필필필필필필필필필필필필필필
# 필필필필필필필필필필필필필필필필필필필필필필필필
# 필필필필필필필필필필필필필필필필필필필필필필필필
# 필필필필필필필필필필필필필필필필필필필필필필필필
# 필필필필필필필필필필필필필필필필필필필필필필필필
# 필필필필필필필필필필필필필필필필필필필필필필필필
# 필필필필필필필필필필필필필필필필필필필필필필필필
# 필필필필필필필필필필필필필필필필필필필필필필필필
# 필필필필필필필필필필필

# WARNING:absl:Found untraced functions such as lstm_cell_layer_call_fn
# , lstm_cell_layer_call_and_return_conditional_losses while saving (showing 
# 2 of 2). These functions will not be directly callable after loading.
