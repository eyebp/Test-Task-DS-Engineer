# I don't know what to put here since this data function doesn't require training,
# And the exploration process is in explore.ipynb.

# let it be just training a gradient boosting model.

import pandas as pd
from xgboost import XGBRegressor
import pickle
import argparse

parser = argparse.ArgumentParser(prog='Training GB model',
                                 description='Just for reference')
parser.add_argument('filename')
args = parser.parse_args()

df = pd.read_csv(args.filename)

# Cast integer type features to categorical type.
for col, t in df.dtypes.items():
    if 'int' in str(t):
        df[col] = df[col].astype('category')

# Split dataset into training and target features.
x, y = df.drop('target', axis=1), df.target

# Train simple tree boosting regression model.
xgreg = XGBRegressor(enable_categorical=True, device='gpu')
xgreg.fit(x, y)

# Save trained model to pickle, just in case.
with open('xgreg.pickle', 'wb') as fw:
    pickle.dump(xgreg, fw)