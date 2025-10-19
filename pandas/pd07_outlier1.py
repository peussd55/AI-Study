import numpy as np
import pandas as pd

aaa = np.array([-10,2,3,4,5,6,7,8,9,10,11,12,50])

def outlier(data):
    q1, q2, q3 = np.percentile(data,[25, 50, 75])
    print('q1 :', q1, '\nq2 :', q2, '\nq3 :', q3)
    iqr = q3 - q1
    print('iqr :', iqr)
    upper = q3 + (1.5*iqr)
    lower = q1 - (1.5*iqr)
    return np.where((data>upper)|(data<lower)), iqr, lower, upper

outlier_loc, iqr, lower, upper = outlier(aaa)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.axhline(upper,xmin=0.4,xmax=0.6, color='red', label='upper_bound')
plt.axhline(lower,xmin=0.4,xmax=0.6, color='red', label='lower_bound')
plt.legend()
plt.show()
