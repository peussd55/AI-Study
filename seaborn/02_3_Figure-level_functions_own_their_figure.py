import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
g = sns.relplot(data = tips, x = 'total_bill', y = 'tip')
g.ax.axline(xy1 = (10,2), slope = .2, color = 'r', dashes = (2,2))
plt.show()