# Scikit-Learn Common Patterns

## Basic pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)
```

## Train/test split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Evaluation
```python
from sklearn.metrics import accuracy_score, classification_report
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

## Common mistakes
- Fitting scaler on full data before split — always fit only on training data
- Not setting random_state — results won't be reproducible
- Using accuracy for imbalanced classes — use F1 or ROC-AUC instead