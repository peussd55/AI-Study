import seaborn as sns
import matplotlib.pyplot as plt

flights = sns.load_dataset('flights')
flights_wide = flights.pivot(index="year", columns="month", values="passengers")

flights_wide_list = [col for _,col in flights_wide.items()]
sns.relplot(data = flights_wide_list, kind='line')

two_series = [flights_wide.loc[:1955, 'Jan'], flights_wide.loc[1952:,"Aug"]]
sns.relplot(data = two_series, kind = 'line')
plt.show()