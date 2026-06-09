from Simple_models import SimpleFeedForward
from TM_models import TemporalConvNet
from environment import FIXED_PARAMS
from environment import sp500_training_pipeline
import pandas as pd
import numpy as np


df = pd.read_csv('data_non_std.csv', parse_dates=['Unnamed: 0'])
df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
features = df.drop(columns=['returns', 'Date']).values.astype(np.float32)
target = df['returns'].values.astype(np.float32)
dates = pd.to_datetime(df['Date'])
tbill3m = df['tbill3m'].values.astype(np.float32)


tcn_arch = {
    'num_channels': [64, 128, 64],
    'pool': 'last'
}

pipe_kwargs = {
    **FIXED_PARAMS,
    'model_kwargs': tcn_arch,
    'model_type': 'tcn'
}
res = sp500_training_pipeline(
    X=features,
    y=target,
    dates=dates,
    tbill3m=tbill3m,
    model_class=TemporalConvNet,
    **pipe_kwargs
)


