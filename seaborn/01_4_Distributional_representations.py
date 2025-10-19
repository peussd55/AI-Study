import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

# sns.displot(data = tips, x = 'total_bill',
#             col = 'time', kde = True)
# sns.displot(data = tips, kind = 'ecdf', x = 'total_bill', 
#             col = 'time', hue = 'smoker', rug = True)
# sns.catplot(data = tips, kind = 'swarm', 
#             x = 'day', y = 'total_bill', hue = 'smoker')
# sns.catplot(data = tips, kind = 'violin', 
#             x = 'day', y = 'total_bill', hue = 'smoker',
#             split = 'True')
sns.catplot(data = tips, kind = 'bar',
            x = 'day', y = 'total_bill',
            hue = 'smoker')


plt.show()


