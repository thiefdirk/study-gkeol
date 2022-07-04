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

#3 sklearn onehotencoder sparse=true 는 매트릭스반환 False는 array 반환 
# 원핫인코딩에서 필요한 것은 array이므로 sparse 옵션에 False를 넣어준다.
# array 가 반환되니 np.argmax 써주면 됨