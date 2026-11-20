import pandas as pd

df = pd.read_csv('digits.csv')

# Add first 5 rows again
df = pd.concat([df, df.head(5)], ignore_index=True)

df.to_csv('digits.csv', index=False)

print("Dataset updated successfully!")