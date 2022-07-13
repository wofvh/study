import numpy as np

a = np.array(range(1,11))
size =5 

def split_x(dataset, size):  
    
    aaa = [] 
    
    for i in range(len(dataset) - size +1):     # for 반복 
       subset = dataset[i : (i +size) ]         # : 범위 
       aaa.append(subset)                       # 더한다.
    return np.array(aaa)

bbb=split_x(a, size)
print(bbb)
print(bbb.shape)
x = bbb[:, :-1]
y = bbb[:, -1]
print(x,y)
print(x.shape, y.shape)

# def split_xy1(dataset, time_steps):                             # def 정의하겠다. 
#   x, y = list(), list()
#   for i in range(len(dataset)):
#     end_number = i + time_steps
#     if end_number > len(dataset) - 1:
#       break
#     tmp_x, tmp_y = dataset[i:end_number], dataset[end_number]
#     x.append(tmp_x)
#     y.append(tmp_y)
#   return np.array(x), np.array(y)

# x, y = split_xy1(dataset, 4)
# print(x, "\n", y)
print(len(a))