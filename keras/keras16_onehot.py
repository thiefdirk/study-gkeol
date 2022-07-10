# [과제]
# 3가지 원핫 인코딩 방식을 비교할것

#1. pandas의 get_dummies https://hongl.tistory.com/89

#2. tensorflow의 to_categorical https://wikidocs.net/22647

#3. sklearn의 OneHotEncoder https://blog.naver.com/PostView.naver?blogId=baek2sm&logNo=221811777148&parentCategoryNo=&categoryNo=75&viewDate=&isShowPopularPosts=false&from=postView

#미세한 차이를 정리하시오!!!

#2 케라스 to_categorical은 0, 1, 2, 3, 4 ... 같이 빠진 순서 없는 숫자를 예측해야 할시에 쓰고
#  만약 0 3 5 7 9 ... 처럼 이가 빠진 놈들도 1 2 4 6 8 의 값들을 다 표현 해줘야 하기 때문에 아웃풋 노드 숫자가 늘어나버림
#  to_categorical을 쓰더라도 위와 같은 경우 1 2 4 6 8 의 컬럼들을 다 drop 시켜주면 사용 할수도 있지만 굳이 그럴바에 딴거 씀

#1 판다스는 아웃풋이 무조건 데이터프레임으로 되기때문에 np.argmax 가 아닌 tf.argmax를 이용
# 겟더미는 인풋을 벡터로 바꿔서 넣어줌
# print(종속.shape)    [50000,1]
# 종속 = pd.get_dummies(종속.reshape(50000)) [이렇게]]
# [50000,1] 이 겟더미 리셰이프를 거치면 [50000,] 이되고 그뒤 겟더미를 거치면 [50000,10]이 됨

#3 sklearn onehotencoder sparse=true 는 매트릭스반환 False는 array 반환 
# 원핫인코딩에서 필요한 것은 array이므로 sparse 옵션에 False를 넣어준다.
# array 가 반환되니 np.argmax 써주면 됨

# 판다스 겟더미가 sklearn 원핫인코더 보다 구린이유
# https://psystat.tistory.com/136
# 대충요약하면 겟더미는 train 에서 학습한 내용중 test 내용에 없는 결과값이 나오면 그걸 인식못함 
# ex) 

# import pandas as pd
# train = pd.DataFrame({'num1':[1,2,3,4,5], 'num2':[10,20,30,40,50], 'cat1':['a', 'a', 'b', 'c', 'c']})
# test = pd.DataFrame({'num1':[1,2,3,4,5], 'num2':[10,20,30,40,50], 'cat1':['a', 'a', 'b', 'b', 'a']})

# 보다시피 train에는 c값이 있는데 test에는 c가 없다
# 겟더미는 이런상황에서 train 에 경우 cat1_a cat1_b cat1_c를 만들어주고 test에서는 cat1_a cat1_b만 만들어 버린다
# 하지만 원핫인코더는 

# fit_transform(train[['cat1']]) 를 이용해서 train에서 학습하고
# transform(test[['cat1']])  으로 test를 학습한 내용을 기억해 변환해줄수 있음

# ###############결론##############
# 1. to_categorical 은 결과값이 이빨안빠지고 0부터 시작하는 형태일때 생각하기 싫으면 써줌
# 2. 겟더미는 결과값이 이빨이 빠진 상태에서 트레인이랑 테스트 값이 안나눠져 있을때, 생각안하고 씌워줌
# 3. 사이킷런 원핫인코더는 제일 골치아픔 이새끼는 벡터입력도 안먹고 매트릭스만 먹기때문에 리셰이프 써줘서 변환도 해줘야되고
#    훈련이랑 변환도 따로 해줘야 되는 상황도 생길수 있음, 그리고 판다스의 시리즈?가 아닌 넘파이 행렬을 이용해줘야됨
#    그래서 아 뭐쓰지 미친 이라는 생각이 들면 아마 이놈 써야할듯
