import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

penguins = sns.load_dataset('penguins')
(
    so.Plot(penguins, x='species', y='body_mass_g')
    .add(so.Bar(), so.Agg())
)

(
    so.Plot(penguins, x='species', y='body_mass_g')
    .add(so.Dot(pointsize = 10), so.Agg())
)

(
    so.Plot(penguins, x='species', y='body_mass_g', color = 'sex')
    .add(so.Dot(pointsize=10), so.Agg())
)

(
    so.Plot(penguins, x='species', y='body_mass_g', color='sex')
    .add(so.Bar(), so.Agg(), so.Dodge())
)

(
    so.Plot(penguins, x='species', y='body_mass_g', color='sex')
    .add(so.Dot(), so.Dodge(), so.Jitter(1))
    .show()
)
