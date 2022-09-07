import numpy as np

f = lambda x: x**2 - 4*x + 6

gradient = lambda x: 2*x - 4

x = -20000
epochs = 20
learning_rate = 0.7

print("step\tx\tf(x)")
print("{:02d}\t{:6.5f}\t{:6.5f}".format(0, x, f(x)))

for i in range(epochs):
    temp = x - learning_rate * gradient(x)
    x = temp
    print("{:02d}\t{:6.5f}\t{:6.5f}".format(i+1, x, f(x)))