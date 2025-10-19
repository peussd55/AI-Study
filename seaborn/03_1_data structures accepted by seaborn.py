import seaborn as sns
import matplotlib.pyplot as plt

flights = sns.load_dataset('flights')
# print(flights.head())
sns.relplot(data = flights, x = 'year', y = 'passengers', hue = 'month', kind = 'line')

flights_wide = flights.pivot(index='year', columns='month', values='passengers')
# print(flights_wide.head)
sns.relplot(data = flights_wide, kind = 'line')

sns.relplot(data = flights, x = 'month', y = 'passengers',
            hue = 'year', kind = 'line')
sns.relplot(data = flights_wide.T, kind = 'line')
sns.catplot(data = flights_wide, kind = 'box')
plt.show()

