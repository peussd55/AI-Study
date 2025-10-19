import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# sns.set_theme()
penguins = sns.load_dataset("penguins")
# sns.relplot(
#      data = penguins,
#      x = "bill_length_mm", y = "bill_depth_mm", hue = "body_mass_g"
# )
# plt.show()

sns.set_theme(style='ticks', font_scale=1.25)
g = sns.relplot(
    data = penguins,
    x = 'bill_length_mm', y = 'bill_depth_mm', hue = 'body_mass_g',
    # hue_norm = (3500,4000),
    palette = 'coolwarm', marker = '*', s=100)
g.set_axis_labels('Bill length (mm)','Bill depth (mm)', labelpad=10)
g.legend.set_title('Body mass (g)')
g.figure.set_size_inches(6.5, 4.5)
g.ax.margins(.15)
g.despine(trim = True)
plt.show()