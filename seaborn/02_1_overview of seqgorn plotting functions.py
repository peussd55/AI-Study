#similar functions for similar tasks
import seaborn as sns
import matplotlib.pyplot as plt

penguins  = sns.load_dataset('penguins')
print(penguins.head())
# sns.histplot(data = penguins, x = 'flipper_length_mm',
#              hue = 'species', multiple = 'stack')
sns.displot(data = penguins, x = 'flipper_length_mm',
            hue = 'species', col = 'species', kind = 'kde')
plt.show()
