import numpy as np
import pandas as pd
from pandas import Series, DataFrame

# Series
series = Series([-1.2, -1.5, -1.7], index=["two", "four", "five"])
print(series.values)
print(series.index)
series2 = series.reindex(['one', 'two', 'three', 'four', 'five'])
print(series2)
series3 = series.reindex(['one', 'two', 'three', 'four', 'five'], method='ffill')
print(series3)
series2 = series2.drop(['one', 'three'])
print(series2)
print(series['two'])
print(series[:2])
print(series[['two', 'three', 'five']])
print(series[series <= -1.5])

# DataFrame
df = DataFrame({'Henry': {'math': 50, 'english': 80},
                'Kate': {'math': 70, 'biology': 40}})
print(df)
print(df['Henry'])
print(df.values)
print(df.index)
print(df.columns)
print('Henry' in df.columns)
index = df.index.append(pd.Index(np.array(['philosopy']), dtype='object'))
df2 = DataFrame(df.values, columns=df.columns, index=pd.Index(np.array(['english', 'math', 'biology'])))
print(df2)
df2 = df2.drop(df2.index)
print(df2)
df2 = DataFrame(np.arange(70, 74).reshape((2,2)),
                index=['engineering', 'magic'],
                columns=df2.columns)
print(df2)
print(df > 70)
df2[df2 < 72] = 0
print(df2)
print(df.loc['biology', df.columns])
print(df.loc[['math', 'english'], df.columns])
print(df.loc[df['Kate'] > 60, df.columns])
df2 = df.reindex(['math', 'magic', 'science'])
print(df2)
df3 = df.reindex(columns=['Henry', 'Shery'], fill_value=0)
print(df3)
df4 = df.reindex(['math', 'magic', 'science'], columns=['Henry', 'Shery'])
print(df.T)
print(df.columns)
print(df.T.columns)

# Series + Series
series1 = Series(np.arange(4))
series2 = Series(np.arange(1, 5))
print(series1 + series2)
print(series1 - series2)

# DataFrame + DataFrame
df1 = DataFrame(np.arange(16).reshape((4, 4)),
                columns=list('abcd'),
                index=np.arange(4))
df2 = DataFrame(np.arange(1, 17).reshape((4, 4)),
                columns=list('bcde'),
                index=np.arange(1, 5))
print(df1 + df2)
print(df1.add(df2, fill_value=0))
print(df1 - df2)
print(df2.sub(df1, fill_value=0))

# DataFrame + Series
df = DataFrame(np.arange(12).reshape((4, 3)),
               columns=list('abc'),
               index=np.arange(4))
series = Series(np.arange(3), index=df.columns)
print(df + series)
series = series.reindex(list('bcd'))
print(df + series)
series = df['a']
print(df.sub(series, axis=0))

# function applying
df = DataFrame({'Henry': {'math': 50, 'english': 80, 'phylosophy': 70, 'biology': 20},
                'Kate': {'math': 70, 'biology': 40, 'english': 90, 'phylosophy': 8},
                'Nancy': {'math': 60, 'biology': 60, 'english': 60, 'phylosophy': 100},
                'Teddy': {'math': 200, 'biology': 0, 'english': 0, 'phylosophy': 60}})
print(df.apply(lambda x: x - 60)) # 縦に適用
df2 = df.T
print(df2.apply(lambda x: np.average(x), axis=1)) # 横に適用
def f(x):
    return Series([x.min(), np.average(x), x.max()], index=['min', 'av.', 'max'])
print(df.apply(lambda x: x.min()))
print(df.apply(f))
print(df2.apply(f).applymap(lambda x: '%.2f' % x))

# Sorting
series = Series(np.random.permutation(np.arange(8)), index=np.arange(8))
result = series.sort_values()
print(result)
df = DataFrame(np.random.permutation(np.arange(16)).reshape((4, 4)),
               columns=np.random.permutation(list('abcd')),
               index=np.random.permutation(np.arange(4)))
result = df.sort_index() # 縦の軸を整列
print(result)
result = df.sort_index(axis=1) # 横の軸を整列
print(result)
result = df.sort_values(by='a')
print(result)