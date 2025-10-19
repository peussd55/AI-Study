import seaborn as sns
import matplotlib.pyplot as plt
import seaborn.objects as so

penguins = sns.load_dataset('penguins')
(
    so.Plot(penguins, x = 'bill_length_mm', y = 'bill_depth_mm',
            color = 'species', pointsize = 'body_mass_g')
    .add(so.Dot())
)

(
    so.Plot(
        penguins, x = 'bill_length_mm', y = 'bill_depth_mm',
        edgecolor = 'sex', edgewidth = 'body_mass_g',
    )
    .add(so.Dot(color='.8'))
)

healthexp = sns.load_dataset('healthexp')
(
    so.Plot(healthexp, x = 'Year', y = 'Life_Expectancy', color = 'Country')
    .add(so.Line())
    .show()
)

