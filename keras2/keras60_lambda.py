gradient1 = lambda x: 2*x - 4 # Lambda : 함수를 한줄로 표현할 수 있음

def gradient2(x):
    temp = 2*x - 4
    return temp


x=3

print(gradient1(x))
print(gradient2(x))