import seaborn as sns
import matplotlib.pyplot as plt

penguins = sns.load_dataset('penguins')

# sns.jointplot(data = penguins, x = 'flipper_length_mm', y = 'bill_length_mm', hue = 'species')
# sns.pairplot(data=penguins, hue = 'species')
sns.jointplot(data = penguins, x = 'flipper_length_mm', y = 'bill_length_mm',
              hue = 'species', kind = 'hist')
plt.show()
