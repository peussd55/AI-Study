import seaborn as sns
import matplotlib.pyplot as plt

fmri = sns.load_dataset('fmri')
# print(fmri.info())
# Data columns (total 5 columns):        
#  #   Column     Non-Null Count  Dtype  
# ---  ------     --------------  -----  
#  0   subject    1064 non-null   object 
#  1   timepoint  1064 non-null   int64  
#  2   event      1064 non-null   object 
#  3   region     1064 non-null   object 
#  4   signal     1064 non-null   float64

sns.relplot(
    data = fmri, kind = 'line',
    x = 'timepoint', y = 'signal', col = 'region',
    hue = 'event', 
    style = 'event',
)
plt.show()