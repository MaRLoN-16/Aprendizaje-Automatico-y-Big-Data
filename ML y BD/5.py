import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame = True).frame

print(housing.head())

housing['income_cat'] = pd.cut(
    housing['MedInc'], 
    bins= {0., 1.5, 3.0, 4.5, 6.0, np.inf}, 
    labels=[1,2,3,4,5]
    )

print(housing.head())

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, stratify = housing['income_cat'], random_state = 42)

values = housing["income_cat"].value_counts(normalize=True)
print(values)

hist = housing["income_cat"].hist()
print(hist)

from sklearn.datasets import fetch_california_housing
dataset = fetch_california_housing(as_frame=True).frame
print(dataset.head())

dataset["income_cat"] = pd.cut(dataset["MedInc"], bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1,2,3,4,5])
print(dataset.head())

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(dataset, test_size=0.2, stratify=dataset["income_cat"], random_state=42)

print(dataset["income_cat"].value_counts())
print(f"Min: {dataset['MedInc'].min()}, Max: {dataset['MedInc'].max()}")

compare = pd.DataFrame({
    "original": dataset['income_cat'].value_counts(normalize=True),
    "train": train_set['income_cat'].value_counts(normalize=True),
    "test": test_set['income_cat'].value_counts(normalize=True),
})
print(compare)

import matplotlib.pyplot as plt
plt.figure(figsize=(12,3))
plt.subplot(1,3,1)
dataset['income_cat'].hist()
plt.title('Original dataset')
plt.subplot(1,3,2)
train_set['income_cat'].hist()
plt.title('Train dataset')
plt.subplot(1,3,3)
test_set['income_cat'].hist()
plt.title('Test dataset')
plt.show()

dataset.drop('income_cat', axis=1, inplace=True)
train_set.drop('income_cat', axis=1, inplace=True)
test_set.drop('income_cat', axis=1, inplace=True)