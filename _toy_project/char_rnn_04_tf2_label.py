import tensorflow as tf
import numpy as np
import os, sys
import time

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

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units, return_sequences=True,
                         stateful=True, kernel_initializer='glorot_uniform'),  # Xavier 정규분포 초기값 설정기
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)  

  # 모델 사용
for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_pred = model(input_example_batch)
  print(example_batch_pred.shape)  
  print(example_batch_pred[0])

  print(input_example_batch)
  print(target_example_batch)
  example_batch_pred = model(input_example_batch)

# 출력은 (batch_size, sequence length, vocab_size)
# 위 예제에서는 시퀀스 길이를 100으로 설정했으면 임의의 길이를 입력해서 모델을 실행할 수 있다.
model.summary()

# 배치의 첫번째 샘플링 시도
sampled_indices = tf.random.categorical(example_batch_pred[0], num_samples=1)
#print(sampled_indices)
sampled_indices = tf.squeeze(sampled_indices, axis=-1)
#print(sampled_indices)

#print(repr(''.join(idx2char[sampled_indices])))
#for pred in example_batch_pred:
#print(np.argmax(pred))
#print(repr(idx2char[np.argmax(pred, axis=-1)]))
#sampled_indices = tf.random.categorical()

# 아직은 훈련되지 않은 모델에 의해 예측된 데이터이다. 이를 알기 쉽게 복호화한다.
print(input_example_batch[0])
print("-----------------------")
print("입력: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("예측된 다음 문자: \n", repr("".join(idx2char[sampled_indices])))

# 모델 훈련: 표준 분류 문제로 다룰 수 있다. 이전 RNN 상태와 이번 타임 스텝의 입력으로 다음 문자의 클래스를 예측합니다.
# 옵티마이저와 손실함수 넣기
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
example_batch_loss = loss(target_example_batch, example_batch_pred)
print("예측 배열 크기(shape): ", example_batch_pred.shape, " # (배치 크기, 시퀀스 길이, 어휘사전 크기)")
print("스칼라 손실:     ", example_batch_loss.numpy().mean())

# 모델 컴파일
model.compile(optimizer='adam', loss=loss) 

# 체크포인트를 사용하여 훈련 중 체크포인트가 저장되도록 합니다.
# 체크포인트가 저장될 폴더
checkpoint_dir = 'D:\study_data\_save/_ModelCheckPoint/char_rnn_project/label'
# 체크 포인트 파일이름
checkpoint_filename = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filename, save_weights_only=True)

# 훈련실행
epochs = 70
keep_training = 1 # 1: 모델을 불러와서 훈련을 계속할 때 0: 훈련을 처음할 때

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
    model = tf.keras.models.load_model('D:\study_data\_save/_ModelCheckPoint/char_rnn_project/label', custom_objects={'loss':loss})

  history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])    
  model.save('D:\study_data\_save/_ModelCheckPoint/char_rnn_project/label')

# 텍스트 생성
# 최근 체크포인트 복원

# 문장 생성을 한다면, 모델을 별도로 만들어야 한다. (배치사이즈를 1로 해야 하기 때문에)
# 이때는 위의 온갓 char2idx, idx2char 등 온갖 변수값들이 필요하기 때문에, 문장생성 코드를 별도 파일로 작성하기가 난감하다.
# 그래서 조건문으로 처리.
if do_generate:
  print("latest_checkpoint: ", tf.train.latest_checkpoint(checkpoint_dir))

  model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
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
  temperature = 1.2

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
  article = generate_text(model, start_string=u"행복 ")
  print(article)

# label epochs 400, temperture 0.4, 100자 생성, 인생
# 인생 
# 지능 평균 지능을 과소평가한다는 것은 있을 수 없다.
# 지도자 지도자는 물과 같이 외유내강(外柔內剛)해야 한다.
# 지도자 남을 따르는 법을 알지 못하는 사람은 좋은 지도자

# label epochs 400, temperture 0.6, 100자 생성, 인생
# 인생 사람의 일이 그에 맞지 않으면 구두의 경우와 흔히 같으니, 너무 크
# 면 비틀거릴 것이요, 너무 작으면 부르틀 것이다.

# label epochs 400, temperture 1.1, 100자 생성, 인생
# 인생 사람의 분노의 폭군에 의존하고 있는 일이 있으니 계산해 보면 부귀
# 한 처지에 있을 때에는 마땅히 빈천한 처지의 고통을 알아야 하고, 젊을 
# 때는 모름지기 노쇠한 처지의 괴로움을 생각해

# label epochs 400, temperture 1.2, 70자 생성, 인생
# 인생 ""예술 가능 사람에게 입을 다물고 있더라도 지금 당장 수행할수 있
# 는 좋은 계획이, 다음주 까지 기다려야 하는 완벽한 계획보다


# label epochs 400, temperture 1.4, 70자 생성, 인생
# 인생 습관을 씨뿌리면 성품이 결실된다. 완성은 휴식은 어리석은 것이 전
# 한 일이 사람, 또는 바구로 낭비하지 않으면 안된다.

# label epochs 700, temperture 1.1, 70자 생성, 인생
# 인생 살 수 있다. 우리는 불행을 자기를 위하여 이용할 수는 있는 것이다
# .  "
# 불화 마음을 합하여 같이 행동할 수 있다고 믿었던

# label epochs 700, temperture 0.9, 130자 생성, 가난
# 가난 못했다고 할 것 같으면 성공의 달성도 필경은 그 인간의 권태의 제
# 물로 만드는데 지나지 않게 된다.  "
# "성공, 근면 내가 성공한 원인은 오직 근면에 있었다. 나는 평생에 단 한
#  번 전쟁이 일어난 이상은, 모든 가능한 방법을 동원하

# label epochs 700, temperture 1.5, 130자 생성, 가족
# 가족 각자 남보다 한 두가지 나은 점은 있지만, 열가지가 다가오면 생겨 보람이요, 마음의 기쁨이다.
