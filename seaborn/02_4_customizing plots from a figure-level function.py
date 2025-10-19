import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
penguins = sns.load_dataset('penguins')
g = sns.relplot(data = penguins, x = 'flipper_length_mm', y = 'bill_length_mm',
                hue = 'sex', col = 'sex')
g.set_axis_labels('flipper length (mm)', 'bill length (mm)')
plt.show()