import matplotlib.pyplot as plt
import seaborn as sns

dots = sns.load_dataset('dots')

# print(dots.info())
# Data columns (total 5 columns):
#  #   Column       Non-Null Count  Dtype  
# ---  ------       --------------  -----  
#  0   align        848 non-null    object 
#  1   choice       848 non-null    object 
#  2   time         848 non-null    int64  
#  3   coherence    848 non-null    float64 
#  4   firing_rate  848 non-null    float64
#  dtypes: float64(2), int64(1), object(2) 

sns.relplot(
    data = dots, 
    # kind = 'line',
    x = 'time', y = 'firing_rate', col = 'align',
    hue = 'choice', size = 'coherence', style = 'choice',
    facet_kws = dict(sharex=False)
)

plt.show()