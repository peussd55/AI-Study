import seaborn as sns
import matplotlib.pyplot as plt

anagrams = sns.load_dataset('anagrams')
print(anagrams.head())

anagrams_long = anagrams.melt(id_vars=['subidr', 'attnr'], var_name = 'solutions', value_name = 'score')
print(anagrams_long.head())

sns.catplot(data=anagrams_long, x = 'solutions', y = 'score', hue = 'attnr',
            kind = 'point')
plt.show()