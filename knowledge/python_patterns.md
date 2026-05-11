# Python Code Patterns

## Function structure
```python
def function_name(param1: type, param2: type = default) -> return_type:
    """
    Brief description.
    
    Args:
        param1: description
        param2: description
    
    Returns:
        description
    
    Raises:
        ValueError: when and why
    """
    if not param1:
        raise ValueError("param1 cannot be empty")
    
    result = ...
    return result
```

## Error handling
```python
try:
    result = risky_operation()
except ValueError as e:
    print(f"Value error: {e}")
    return None
except Exception as e:
    raise RuntimeError(f"Unexpected error: {e}") from e
```

## Common mistakes to avoid
- Mutable default arguments: def f(x=[]) — use def f(x=None): if x is None: x = []
- Bare except: — always catch specific exceptions
- Not closing files — always use with open() as f:
- Global variables — pass as parameters instead