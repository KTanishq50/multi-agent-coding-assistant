# Pandas Common Patterns

## Loading data
```python
import pandas as pd
df = pd.read_csv('file.csv')
df = pd.read_csv('file.csv', index_col=0, parse_dates=['date_col'])
```

## Inspecting data
```python
df.head()
df.info()
df.describe()
df.shape
df.columns
df.dtypes
```

## Selecting data
```python
df['column']           # single column → Series
df[['col1', 'col2']]   # multiple columns → DataFrame
df.loc[0:5, 'col']     # label-based
df.iloc[0:5, 0]        # position-based
df[df['col'] > 5]      # boolean filtering
```

## Common operations
```python
df.groupby('col').mean()
df.merge(df2, on='key')
df.fillna(0)
df.dropna()
df.sort_values('col', ascending=False)
```

## Common mistakes
- Chained indexing df['a']['b'] — use df.loc[:, 'b'] instead
- Modifying a slice — always use .copy() when slicing
- String operations on non-string columns — use .astype(str) first