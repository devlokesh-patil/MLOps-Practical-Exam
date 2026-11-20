import pandas as pd
from sklearn.datasets import load_digits

data = load_digits()

df = pd.DataFrame(data.data)
df['target'] = data.target

df.to_csv('digits.csv', index=False)

print("Dataset created successfully!")