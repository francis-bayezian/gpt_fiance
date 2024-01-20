
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Read the parquet file
data = pd.read_parquet('daily.pq')

# Reset the index
df_reset = data.reset_index()

# Fill forward the 'symbol' column
change_index = df_reset['symbol'].ne(df_reset['symbol'].shift())
df_reset['symbol'] = df_reset['symbol'].ffill()

# Create subset dataframes for TSLA and SPY
df_tsla = df_reset[df_reset['symbol'] == 'TSLA']
df_spy = df_reset[df_reset['symbol'] == 'SPY']

# Merge the two dataframes on 'timestamp'
merged_df = pd.merge(df_tsla, df_spy, on='timestamp', suffixes=('_tsla', '_spy'))

# Calculate returns for TSLA and SPY
merged_df['return_tsla'] = merged_df['close_tsla'].pct_change()
merged_df['return_spy'] = merged_df['close_spy'].pct_change()

# Drop the first row (which has NaN returns)
merged_df = merged_df.iloc[1:]

# Fit a linear regression model
X = merged_df[['return_spy']]
y = merged_df['return_tsla']
model = LinearRegression().fit(X, y)

# Plot the regression line and the actual data
plt.scatter(merged_df['return_spy'], merged_df['return_tsla'], color='blue')
plt.plot(merged_df['return_spy'], model.predict(X), color='red')
plt.xlabel('SPY Returns')
plt.ylabel('TSLA Returns')
plt.title('Linear Regression of TSLA vs SPY Returns')
plt.savefig('regression_plot.png')
plt.show()
