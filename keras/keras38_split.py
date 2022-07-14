import numpy as np

a = np.array(range(1,11))
size =5 

def split_x(dataset, size):  
    
    aaa = [] 
                                                # aaa = [] 빈값 설정.
    for i in range(len(dataset) - size +1):     # for 반복횟수 range(len)= 10  - size(5) + 1 
       subset = dataset[i : (i +size) ]         # i = 시작값 : 시작값+5  1:5 
       aaa.append(subset)                       # aaa []빈값에 subset값을 append(더해준다)  
    return np.array(aaa)                        # aaa를 반환할거다.

bbb=split_x(a, size)                            #bbb는 split_x로 a 값과 size을 계산한다.
print(bbb)                                      
print(bbb.shape)
x = bbb[:, :-1]                                 # x= 전체에서 마지막행 빼고 
y = bbb[:, -1]                                  # y= 전체에서 마지막행 만
print(x,y)
print(x.shape, y.shape)

# def split_xy1(dataset, time_steps):           # def 정의하겠다. 
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
