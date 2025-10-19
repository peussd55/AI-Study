import seaborn as sns
import matplotlib.pyplot as plt

flights = sns.load_dataset('flights')
flights_dict = flights.to_dict()
sns.relplot(data = flights_dict, x='year', y='passengers',
            hue = 'month', kind = 'line')
flights_avg = flights.groupby('year').mean(numeric_only = True)
sns.relplot(data=flights_avg, x='year', y='passengers', kind='line')
plt.show()