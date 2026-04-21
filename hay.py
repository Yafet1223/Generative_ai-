import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# example data: study hours → pass/fail
df = pd.DataFrame({
    "Hours": [1,2,3,4,5,6,7,8],
    "Pass":  [0,0,0,0,1,1,1,1]
})

X = df[["Hours"]]
y = df["Pass"]

model = LogisticRegression()
model.fit(X, y)

# predict probability
print(model.predict_proba([[5.5]]))

# final class
print(model.predict([[5.5]]))