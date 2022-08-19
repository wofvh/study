import numpy as np
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
               [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])
aaa = np.transpose(aaa)


print(aaa)
print(aaa[:,0])
print(aaa[:,1])




def outliers(data_out, i) :
    quartile_1, q2, quartile_3 = np.percentile(data_out[:,i],
                                               [25,50,75])
    print('1사분위 :',quartile_1)               # 1사분위 : 4.0
    print('q2:',q2)                             # q2: 7.0  (중위값)
    print('3사분위 :',quartile_3)               # 3사분위 : 10.0
    iqr = quartile_3 - quartile_1
    print('iqr:', iqr)                          # iqr: 6.0
    lower_bound = quartile_1 -(iqr *1.5)
    upper_bound = quartile_3 +(iqr *1.5)
    print('upper_bound:',upper_bound)           # upper_bound: 19.0
    print('lower_bound:',lower_bound)           # lower_bound: -5.0
    return np.where((data_out[:,i]>upper_bound)|     # aaa에서 upper_bound(19.0)보다 높은 번째 수 8
                    (data_out[:,i]<lower_bound))     # aaa에서 lower_bound(-5.0)보다 높은 번째 수 2
  
outliers_loc = outliers(aaa, 0) #outliers(data_out, i)
outliers_loc2 = outliers(aaa,1)

print('이상치[1]의 위치:',outliers_loc)            # 이상치의 위치: (array([0, 12], dtype=int64),
print('이상치[2]의 위치:',outliers_loc2)            # 이상치의 위치: (array([0, 12], dtype=int64),
    
import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()


# -70000 -30 -10 ,2,3,4,5,6,7,8,9, 10,11,12,50,100,200,210,350,400,420,500,600,800,900,1000
#    0    1   2   3 4 5 6 7 8 9 10 11 12 13 14 15  16  17  18  19  20  21  22  23  24  25