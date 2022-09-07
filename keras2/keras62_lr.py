x = 10
y = 10
w = 1111
lr = 0.001
epochs = 300000

for i in range(epochs):
    predict = w * x
    loss = (predict - y) ** 2
    
    print("loss : ", round(loss, 4), "\tpredict : ", round(predict, 4))
    
    up_predict = w * (x + lr)
    up_loss = (up_predict - y) ** 2
    
    down_predict = w * (x - lr)
    down_loss = (down_predict - y) ** 2
    
    if up_loss > down_loss:
        w = w - lr
    else:
        w = w + lr