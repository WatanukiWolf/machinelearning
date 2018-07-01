import numpy as np
import pandas as pd
import seaborn as sns

x = np.random.normal(size=100)

titanic = sns.load_dataset("titanic")
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")

sns.distplot(x, kde=False, rug=False, bins=10)
sns.jointplot('sepal_width', 'petal_length', data=iris)
sns.pairplot(iris)
sns.pairplot(iris, hue='species')