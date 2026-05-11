# NumPy Common Patterns

## Creating arrays
```python
import numpy as np
arr = np.array([1, 2, 3])
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
arange = np.arange(0, 10, 2)
```

## Array operations
```python
# dot product
result = np.dot(a, b)
# or using @
result = a @ b

# sum, mean, std
np.sum(arr)
np.mean(arr)
np.std(arr)

# reshape
arr.reshape(3, 2)
arr.flatten()
```

## Indexing and slicing
```python
arr[0]        # first element
arr[-1]       # last element
arr[1:4]      # slice
arr[arr > 5]  # boolean indexing
```

## Common mistakes
- Using Python lists where numpy arrays expected — convert with np.array()
- Forgetting axis parameter in sum/mean — np.sum(arr, axis=0) for column sum
- Shape mismatch in dot product — (m,n) @ (n,p) works, (m,n) @ (m,p) fails