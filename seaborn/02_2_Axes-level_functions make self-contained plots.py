import seaborn as sns
import matplotlib.pyplot as plt

penguins = sns.load_dataset('penguins')
f, axs = plt.subplots(1,2, figsize = (8,4), gridspec_kw = dict(width_ratios = [4,3]))
sns.scatterplot(data = penguins, x = 'flipper_length_mm', y = 'bill_length_mm',
                hue = 'species', ax = axs[0])
sns.histplot(data = penguins, x = 'species', hue = 'species', shrink = .8,
             legend = False, ax = axs[1])
f.tight_layout()
plt.show()