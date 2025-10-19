import seaborn as sns
import matplotlib.pyplot as plt
####### apply the default theme
sns.set_theme()

####### load an example dataset
tips = sns.load_dataset('tips')
# print(tips.info())
#  #   Column      Non-Null Count  Dtype
# ---  ------      --------------  -----
#  0   total_bill  244 non-null    float64
#  1   tip         244 non-null    float64
#  2   sex         244 non-null    category
#  3   smoker      244 non-null    category
#  4   day         244 non-null    category
#  5   time        244 non-null    category
#  6   size        244 non-null    int64

# [244 rows x 7 columns]

###### create a visualization
sns.relplot(
    data = tips, x = 'total_bill', y = 'tip', col = 'time',
    hue = 'smoker', style = 'smoker', size = 'size',)
plt.show()