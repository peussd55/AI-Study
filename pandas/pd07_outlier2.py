### <<42>>

import numpy as np 
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
               [100,200,-30,400,500,600,-7000,800,900,1000,210,420,350]]
               ).T
print(aaa.shape)    # (13, 2)

# outlier 찾기
aaa_1 = np.sort(aaa[:,0])
print(aaa_1)    
# [-10   2   3   4   5   6   7   8   9  10  11  12  50]
aaa_2 = np.sort(aaa[:,1])
print(aaa_2)
# [-7000    -30    100    200    210    350    400    420    500    600
#     800    900   1000]

def outlier(data):
    # sorted_columns = [np.sort(aaa[:,i]) for i in range(aaa.shape[1])]
    
    quartile_1, q2, quartile_3 = np.percentile(data, [25,50,75])
    print('1사분위 : ', quartile_1)
    print('2사분위 : ', q2)    
    print('3사분위 : ', quartile_3)
    iqr = quartile_3 - quartile_1
    print('IQR : ', iqr)
    lower_bound = quartile_1 - (iqr*1.5)
    upper_bound = quartile_3 + (iqr*1.5)
     
    # np.where : 위치를 반환
    return np.where((data > upper_bound) | (data < lower_bound)), \
        iqr, lower_bound, upper_bound

outlier_loc_1, iqr_1, low_1, up_1 = outlier(aaa_1)
outlier_loc_2, iqr_2, low_2, up_2 = outlier(aaa_2)

print('_1, 이상치의 위치 : ', outlier_loc_1)  # 이상치의 위치 :  (array([ 0, 12], dtype=int64),)

import matplotlib.pyplot as plt 
# plt.boxplot(aaa_2)
# plt.axhline(up_2, color='red', label='upper bound')
# plt.axhline(low_2, color='pink', label='lower bound')
# plt.legend()
# plt.show()

# 서브플롯 생성: 1행 2열(subplots)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# 첫 번째 박스플롯
axes[0].boxplot(aaa_1)
axes[0].axhline(up_1, color='red', label='upper bound')
axes[0].axhline(low_1, color='pink', label='lower bound')
axes[0].set_title('Boxplot of aaa_1')
axes[0].legend()
# 두 번째 박스플롯
axes[1].boxplot(aaa_2)
axes[1].axhline(up_2, color='red', label='upper bound')
axes[1].axhline(low_2, color='pink', label='lower bound')
axes[1].set_title('Boxplot of aaa_2')
axes[1].legend()
plt.tight_layout()
plt.show()
