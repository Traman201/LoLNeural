from unicodedata import name
from pandas import DataFrame
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


fileName = "YDataset.csv"
colnames = ["time",
        "kill",
        "death",
        "assist",
        "abilityPower",
        "armor",
        "armorPenPercent",
        "attackDamage",
        "attackSpeed",
        "ccReduction",
        "health",
        "healthMax",
        "healthRegen",
        "lifesteal",
        "magicPen",
        "magicResist",
        "movementSpeed",
        "omnivamp",
        "power",
        "powerMax",
        "powerRegen","y"]
Dataset = pd.read_csv(fileName)
stats = Dataset.describe(include = 'all')

el = "death"

st = dict(stats[el])
fig = plt.figure(figsize=(10, 10), dpi= 80)
grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)
ax_main = fig.add_subplot(grid[:-1, :-1])
ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])
sns.boxplot(Dataset[el], ax=ax_bottom, orient="h",showmeans=True)
#Dataset[el].plot.kde(ax=ax_main, xlim = (st["min"] , st["max"]))
Dataset[el].plot.hist(density=True, ax=ax_main, xlim = (st["min"] ,st["max"]))
print("mod ",Dataset[el].mode())
print("median ",Dataset[el].median())
print("dispersion " ,Dataset.var()[el])
print("Asуmьetry ",Dataset[el].skew())
print("axcess ",Dataset[el].kurt())
plt.show()