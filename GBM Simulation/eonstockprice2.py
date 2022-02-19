import pandas as pd
import numpy as np
import nasdaqdatalink
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

start_date = '2019-07-01'
end_date = '2019-07-31'
pred_end_date = '2019-08-31'

# We get daily closing stock prices of E.ON for July 2019
S_eon = nasdaqdatalink.get("FSE/EON_X",
               start_date = start_date, end_date = end_date
               ).reset_index(drop = False)[['Date', 'Close']]
               

# Parameters
So = S_eon.loc[S_eon.shape[0] - 1, "Close"]
dt = 1
n_of_wkdays = pd.date_range(start = pd.to_datetime(end_date, 
              format = "%Y-%m-%d") + pd.Timedelta('1 days'), 
              end = pd.to_datetime(pred_end_date, 
              format = "%Y-%m-%d")).to_series(
              ).map(lambda x: 
              1 if x.isoweekday() in range(1,6) else 0).sum()
T = n_of_wkdays
N = T / dt
t = np.arange(1, int(N) + 1)

returns = (S_eon.loc[1:, 'Close'] - \
          S_eon.shift(1).loc[1:, 'Close']) / \
          S_eon.shift(1).loc[1:, 'Close']
print(returns.tolist())
