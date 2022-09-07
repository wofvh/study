import numpy as np
f = lambda x : x**2 -4*x +6

# def f(x):
#     temp = x**2 -4*x + 6
#     return temp

gradient = lambda x :2*x - 4

x =30.0 # 초기값
epoch =20 
learning_rate = 0.25

print('step\t x\t f(x)')
print("{:02d}\t {:6.5f}\t {:6.5f}\t".format(0,x,f(x)))

for i in range(epoch):
    x = x -learning_rate * gradient(x) # gradient(x)= 미분값 
     
    print("{:02d}\t {:6.5f}\t {:6.5f}\t".format(i+1,x,f(x)))




    
    