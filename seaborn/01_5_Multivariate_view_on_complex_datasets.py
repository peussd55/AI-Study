import seaborn as sns
import matplotlib.pyplot as plt

penguins = sns.load_dataset('penguins')
# print(penguins.info())
#  #   Column             Non-Null Count  Dtype
# ---  ------             --------------  -----
#  0   species            344 non-null    object
#  1   island             344 non-null    object
#  2   bill_length_mm     342 non-null    float64
#  3   bill_depth_mm      342 non-null    float64
#  4   flipper_length_mm  342 non-null    float64
#  5   body_mass_g        342 non-null    float64
#  6   sex                333 non-null    object

# sns.jointplot(data = penguins, x = 'bill_length_mm', y = 'bill_depth_mm',
#               hue = 'species')
sns.pairplot(data = penguins, hue = 'species')
plt.show()