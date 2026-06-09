# Selected Quantitative Tools: S&P 500 Return Forecasting

This repository contains an UZH Selected Quantitative Tools course project on S&P 500 return prediction. The study compares six neural forecasting architectures under rolling and expanding time-series validation, with and without autoencoder preprocessing. The objective is not only to find the lowest-error model, but also to understand how different neural architectures behave on noisy financial time series: which models produce stable forecasts, which react to regime changes, and how feature compression changes the forecast distribution.

The written project document is [`SQT-3.pdf`](SQT-3.pdf), titled *Investments: Machine Learning for Finance* by Konrad Ochedzan, Aron Miklos, and Monna Dimitrova. It is the stored write-up of the study: motivation, architecture descriptions, experimental setup, result tables, model-specific interpretation, and suggested future work.

## Research Setup

The supervised learning task is one-step-ahead S&P 500 return forecasting. For each date $t$, the data contains a target return

```math
y_t \in \mathbb{R}
```

and a feature vector

```math
x_t \in \mathbb{R}^{22}.
```

The feature set combines market, macroeconomic, technical, volatility, commodity, FX, international index, and Google Trends variables. The model forecasts

```math
\hat{y}_t = f_\theta(\mathcal{I}_t),
```

where $\mathcal{I}_t$ is the information set available before predicting $y_t$. For non-sequential models this is mainly the current feature vector plus autoregressive return lags; for sequential models it is a window of past observations.

Two input regimes are tested:

```math
\text{raw features: } \tilde{x}_t,
\qquad
\text{autoencoder features: } z_t = E_\phi(\tilde{x}_t).
```

The six forecasting architectures are FeedForward, LSTM, Transformer, CNN, Temporal Convolutional Network (TCN), and Temporal Fusion Transformer (TFT). Each is trained under both rolling and expanding validation.

## Data

[`data_non_std.csv`](data_non_std.csv) is the main modeling table. It contains 4,104 daily observations from February 2009 onward. The target is `returns`; explanatory variables include `vix`, `tbill3m`, `term_spread`, `nikkei`, `ftse`, `hsi`, `crude`, `gold`, `silver`, `gas`, FX rates, `rsi`, `macd_diff`, `gtrends`, `unemployment`, `cpi`, and `realized_vol`.

Features are standardized inside each fold, using only the training sample:

```math
\tilde{x}_{t,j}
=
\frac{x_{t,j}-\mu^{train}_j}{\sigma^{train}_j}.
```

The fitted training scaler is then applied to the corresponding test fold. This is essential: using a global scaler would leak future distributional information into the training stage.

## Validation

The study uses chronological validation rather than random cross-validation. This matches the forecasting problem: train on the past, test on the future.

For a rolling window, the training window has fixed length and moves forward:

```math
\mathcal{T}^{roll}_k
=
\{t:s_k-W_{train}\le t<s_k\},
\qquad
\mathcal{V}_k
=
\{t:s_k\le t<s_k+W_{test}\}.
```

For an expanding window, the training set starts at the first observation and grows:

```math
\mathcal{T}^{exp}_k
=
\{t:t_0\le t<s_k\},
\qquad
\mathcal{V}_k
=
\{t:s_k\le t<s_k+W_{test}\}.
```

The main configuration uses $W_{train}=3$ years and $W_{test}=1$ year. Rolling windows emphasize recent market regimes; expanding windows emphasize maximal historical information.

## Model Mathematics

### Input Construction

Let $p$ be the number of autoregressive return lags and $L$ the sequence length. For the FeedForward model the input is

```math
u_t
=
\left[
\tilde{x}_t,\,
y_{t-1},\ldots,y_{t-p}
\right].
```

For sequence models, the input tensor is

```math
U_t
=
\left[
u_{t-L},u_{t-L+1},\ldots,u_{t-1}
\right]
\in \mathbb{R}^{L\times d},
```

where each $u_\tau$ contains features and padded autoregressive return information. The forecast is produced from this historical block.

### Autoencoder Preprocessing

The autoencoder tests whether a lower-dimensional learned representation improves forecasting. The encoder maps standardized features to latent factors,

```math
z_t = E_\phi(\tilde{x}_t),
\qquad
z_t\in\mathbb{R}^{q},
```

with $q=10$ in the main experiments. The decoder reconstructs the standardized feature vector:

```math
\hat{x}_t = D_\psi(z_t).
```

The autoencoder is trained only on the training fold:

```math
\min_{\phi,\psi}
\frac{1}{n}
\sum_{t\in\mathcal{T}}
\left\|
\tilde{x}_t
-
D_\psi(E_\phi(\tilde{x}_t))
\right\|_2^2.
```

After training, only $E_\phi$ is used. The forecasting model receives $z_t$ instead of $\tilde{x}_t$. Mathematically, this replaces the original feature space by a learned nonlinear factor space. Qualitatively, it can remove noise and redundancy, but it can also suppress useful extreme-signal variation.

### Supervised Loss and Regularization

The main supervised objective is mean squared error:

```math
\mathcal{L}_{MSE}(\theta)
=
\frac{1}{n}
\sum_{t\in\mathcal{T}}
(y_t-\hat{y}_t)^2.
```

The pipeline also supports ElasticNet regularization:

```math
\mathcal{L}(\theta)
=
\mathcal{L}_{MSE}(\theta)
+
\alpha
\left(
\rho\|\theta\|_1
+
(1-\rho)\|\theta\|_2^2
\right).
```

The L1 term encourages sparse weights and therefore implicit feature selection. The L2 term shrinks weights continuously, reducing sensitivity to single noisy variables. This matters in financial prediction because many indicators are correlated and their relationships with returns are unstable.

The code also contains a Sharpe-ratio loss experiment. It converts forecasts into positions

```math
p_t=\tanh(\hat{y}_t),
```

strategy returns

```math
r^{strat}_t=p_ty_t,
```

and minimizes negative annualized Sharpe:

```math
\mathcal{L}_{Sharpe}
=
-
\frac{
\mathbb{E}[r^{strat}_t-r_f/252]
}{
\mathrm{Std}(r^{strat}_t-r_f/252)+\varepsilon
}.
```

The final experiments use MSE as the training loss because direct Sharpe optimization was less stable.

### FeedForward Network

The FeedForward model is a nonlinear regression baseline. It receives $u_t$ and applies two hidden layers:

```math
h_1
=
\mathrm{ReLU}(W_1u_t+b_1),
```

```math
h_2
=
\mathrm{ReLU}(W_2h_1+b_2),
```

```math
\hat{y}_t
=
W_3h_2+b_3.
```

Dropout is applied between layers:

```math
\tilde{h}_\ell
=
m_\ell\odot h_\ell,
\qquad
m_{\ell,i}\sim\mathrm{Bernoulli}(1-r),
```

where $r$ is the dropout rate. This model can learn nonlinear interactions between contemporaneous features and lagged returns, but it has no internal memory. All time dependence must be explicitly placed into $u_t$ through lagged variables.

### LSTM

The LSTM is designed for sequential dependence. For each time step in $U_t$, it updates a hidden state $h_\tau$ and cell state $c_\tau$. The gates are

```math
f_\tau
=
\sigma(W_f[u_\tau,h_{\tau-1}]+b_f),
```

```math
i_\tau
=
\sigma(W_i[u_\tau,h_{\tau-1}]+b_i),
```

```math
o_\tau
=
\sigma(W_o[u_\tau,h_{\tau-1}]+b_o),
```

and the candidate memory is

```math
\tilde{c}_\tau
=
\tanh(W_c[u_\tau,h_{\tau-1}]+b_c).
```

The cell and hidden states update as

```math
c_\tau
=
f_\tau\odot c_{\tau-1}
+
i_\tau\odot\tilde{c}_\tau,
```

```math
h_\tau
=
o_\tau\odot\tanh(c_\tau).
```

The final forecast uses the last hidden state:

```math
\hat{y}_t
=
g_\theta(h_{t-1}).
```

The forget gate $f_\tau$ controls persistence of old information, the input gate $i_\tau$ controls entry of new information, and the output gate $o_\tau$ controls what part of memory becomes visible to the prediction head. This makes the LSTM suitable for momentum, mean-reversion, and volatility-regime patterns that depend on recent history.

### Transformer

The Transformer uses attention rather than recurrence. Each input vector is projected into a model dimension:

```math
e_\tau = W_eu_\tau+b_e.
```

Since attention alone has no natural ordering, sinusoidal positional encodings are added:

```math
PE_{pos,2i}
=
\sin\left(\frac{pos}{10000^{2i/d}}\right),
\qquad
PE_{pos,2i+1}
=
\cos\left(\frac{pos}{10000^{2i/d}}\right).
```

The sequence representation is

```math
X_\tau=e_\tau+PE_\tau.
```

For one attention head,

```math
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
```

and scaled dot-product attention is

```math
A(Q,K,V)
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
```

Multi-head attention is

```math
\mathrm{MHA}(X)
=
\mathrm{Concat}(A_1,\ldots,A_H)W_O.
```

Each encoder layer applies attention, residual connections, normalization, and a pointwise feed-forward block:

```math
X'
=
\mathrm{LayerNorm}(X+\mathrm{MHA}(X)),
```

```math
X''
=
\mathrm{LayerNorm}(X'+FFN(X')).
```

The final prediction uses the encoded last time step. This architecture can directly compare all dates in the input window, so it can learn nonlocal temporal relationships. Its weakness in this study is sensitivity to the amount and type of historical data.

### Convolutional Neural Network

The CNN treats each feature sequence as a one-dimensional signal. A convolutional filter produces

```math
z_{c,\tau}
=
b_c
+
\sum_{j=1}^{d}
\sum_{k=0}^{K-1}
w_{c,j,k}u_{\tau-k,j}.
```

With activation and pooling,

```math
a_{c,\tau}
=
\mathrm{ReLU}(z_{c,\tau}),
\qquad
m_{c,r}
=
\max_{\tau\in B_r} a_{c,\tau}.
```

The implementation stacks several convolutional layers, applies dropout, then uses adaptive average pooling:

```math
\bar{m}_c
=
\frac{1}{L'}
\sum_{\tau=1}^{L'} a_{c,\tau}.
```

The forecast is linear in the pooled channels:

```math
\hat{y}_t
=
w^\top\bar{m}+b.
```

The CNN is useful for local temporal motifs: short bursts of volatility, short-term trend continuation, or local reversal patterns. It is less naturally suited to very long-range dependencies unless many layers or larger receptive fields are used.

### Temporal Convolutional Network

The TCN is a causal convolutional model with dilation. Causality means the forecast at time $t$ never depends on future observations. A dilated convolution is

```math
z_\tau
=
\sum_{k=0}^{K-1}
w_k u_{\tau-dk},
```

where $d$ is the dilation factor. In the implementation, dilation grows by layer:

```math
d_\ell=2^\ell.
```

This gives a wide receptive field without requiring recurrence. A temporal block applies two dilated causal convolutions, nonlinearities, normalization, dropout, and a residual connection:

```math
F_\ell(x)
=
\mathrm{Dropout}
\left(
\mathrm{GroupNorm}
\left(
\mathrm{ReLU}
\left(
\mathrm{Conv}^{d_\ell}(x)
\right)
\right)
\right),
```

```math
\mathrm{Block}_\ell(x)
=
\mathrm{ReLU}(F_\ell(F_\ell(x))+R_\ell(x)).
```

Here $R_\ell(x)=x$ if the dimensions match and otherwise a $1\times1$ convolution. This model is designed to preserve temporal order, avoid future leakage, and learn both short- and medium-range temporal structure.

### Temporal Fusion Transformer

The implemented TFT is a simplified hybrid architecture combining feature selection, recurrent processing, static enrichment, and attention. It first maps raw inputs to selected hidden features:

```math
v_\tau = V_\theta(u_\tau).
```

The code uses a learned variable-selection network rather than a fixed hand-selected subset. In principle this lets the model assign different effective weights to indicators depending on their predictive usefulness.

Static context is approximated by the average hidden representation over the window:

```math
s
=
\frac{1}{L}
\sum_{\tau=t-L}^{t-1}v_\tau,
\qquad
\bar{s}=W_ss+b_s.
```

The enriched temporal input is

```math
\tilde{v}_\tau=v_\tau+\bar{s}.
```

An LSTM processes the enriched sequence:

```math
h_\tau^{LSTM}
=
\mathrm{LSTM}(\tilde{v}_\tau,h_{\tau-1}^{LSTM}).
```

Multi-head attention then fuses temporal information:

```math
a_\tau
=
\mathrm{MHA}
(h_\tau^{LSTM}).
```

The final layers use residual normalization and a feed-forward block:

```math
r_\tau
=
\mathrm{LayerNorm}(a_\tau+h_\tau^{LSTM}),
```

```math
q_\tau
=
\mathrm{LayerNorm}(FFN(r_\tau)+r_\tau),
```

```math
\hat{y}_t=g_\theta(q_{t-1}).
```

This model can combine sequential memory from the LSTM with direct temporal comparison from attention. In the empirical results, this matters because TFT forecasts without autoencoder preprocessing are more reactive to large market moves than the smoother low-MSE models.

## Evaluation Metrics

Point-forecast quality is measured by

```math
MAE
=
\frac{1}{n}\sum_{t=1}^{n}|y_t-\hat{y}_t|,
```

```math
RMSE
=
\sqrt{
\frac{1}{n}\sum_{t=1}^{n}(y_t-\hat{y}_t)^2
},
```

```math
R^2
=
1-
\frac{\sum_t(y_t-\hat{y}_t)^2}
{\sum_t(y_t-\bar{y})^2}.
```

The directional hit rate is

```math
HR
=
\frac{1}{n}
\sum_{t=1}^{n}
\mathbf{1}
\left[
\mathrm{sign}(\hat{y}_t)
=
\mathrm{sign}(y_t)
\right].
```

Trading diagnostics use position sizing

```math
p_t
=
\tanh
\left(
\frac{\hat{y}_t}{2\hat{\sigma}_{20}}
\right),
\qquad
r_t^{strat}=p_ty_t.
```

The annualized Sharpe ratio is

```math
SR
=
\sqrt{252}
\frac{
\mathbb{E}[r_t^{strat}-r_f/252]
}{
\mathrm{Std}(r_t^{strat})
}.
```

The annualized Sortino ratio replaces total volatility by downside volatility:

```math
Sortino
=
\sqrt{252}
\frac{\mathbb{E}[r_t^{strat}-r_f/252]}{\mathrm{Std}\left((r_t^{strat}-r_f/252)\mathbf{1}_{r_t^{strat}<r_f/252}\right)}.
```

Forecast differences are tested with the Diebold-Mariano statistic. For two models with errors $e_{1,t}=y_t-\hat{y}_{1,t}$ and $e_{2,t}=y_t-\hat{y}_{2,t}$, define squared-error loss differential

```math
d_t=e_{1,t}^2-e_{2,t}^2.
```

The test statistic is

```math
DM
=
\frac{\bar{d}}
{\sqrt{\widehat{\mathrm{Var}}(\bar{d})}},
\qquad
\bar{d}
=
\frac{1}{n}\sum_{t=1}^{n}d_t.
```

The variance estimate uses autocovariances of $d_t$, corresponding to a Newey-West/HAC correction for forecast horizon $h$:

```math
\widehat{\mathrm{Var}}(\bar{d})
=
\frac{1}{n}
\left(
\hat{\gamma}_0
+2\sum_{k=1}^{h-1}\hat{\gamma}_k
\right).
```

Small p-values mean the two models have statistically distinguishable forecast losses.

## Results

The strongest model by average out-of-sample MSE is LSTM with autoencoder preprocessing. It is best in both rolling and expanding validation. The main MSE comparison, shown as $MSE\times 10^3$, is:

| Architecture | Rolling | Expanding | Rolling - Expanding |
|---|---:|---:|---:|
| LSTM + Autoencoder | 0.092 | 0.090 | 0.002 |
| LSTM | 0.226 | 0.126 | 0.100 |
| Transformer + Autoencoder | 0.106 | 0.100 | 0.006 |
| Transformer | 0.569 | 0.111 | 0.458 |
| TFT + Autoencoder | 0.111 | 0.115 | -0.004 |
| TFT | 0.131 | 0.223 | -0.092 |
| FeedForward + Autoencoder | 0.160 | 0.123 | 0.037 |
| FeedForward | 5.900 | 0.323 | 5.577 |
| CNN + Autoencoder | 0.216 | 0.158 | 0.058 |
| CNN | 1.521 | 0.143 | 1.378 |

Autoencoder preprocessing materially improves the simpler models in rolling validation. FeedForward improves from $5.900$ to $0.160$, and CNN improves from $1.521$ to $0.216$, both in $MSE\times 10^3$. This supports the interpretation that autoencoder features remove noise and redundant indicators that simple architectures cannot filter internally.

For advanced sequence models, the effect is more nuanced. LSTM + Autoencoder has the lowest MSE, but its forecasts are comparatively smooth and close to the long-run mean. Transformer + Autoencoder is also much more stable than the raw Transformer in rolling validation. TFT behaves differently: the raw TFT has worse MSE than the autoencoder TFT in expanding validation, but it is more responsive to extreme market conditions and produces a broader range of predictions.

Hit-rate results add a second view of performance. In expanding validation, raw TFT reaches the strongest test hit rate, approximately $58.4\%$, while FeedForward, LSTM, and their autoencoder variants are mostly in the low-to-mid $50\%$ range. This matters because low MSE and useful directional classification are not identical objectives. A model can reduce squared error by staying close to zero, but such a model may miss profitable directional moves.

The Sharpe-ratio results are weaker than the point-forecast results. Average out-of-sample Sharpe ratios are negative across the final model set, mainly because several folds have large losses and high fold-to-fold variance. Many individual folds still show positive Sharpe, so the result should be read as evidence that direct return prediction is not automatically a complete trading strategy. Position sizing, risk control, transaction costs, and model mixing would need to be handled separately.

The Diebold-Mariano tests show that most model pairs have statistically different forecast errors under squared-error loss. This is an important result: the architectures are not merely different implementations of the same forecast. They learn different approximations to the return-generating structure. The practical implication is that model choice should depend on the intended objective: stable low-error prediction, directional sensitivity, or reactivity to market stress.

The central conclusion is therefore a trade-off. LSTM + Autoencoder is the best statistical forecaster by MSE. TFT without autoencoder is more informative about volatility and extreme regimes. The natural extension is a mixture model

```math
\hat{y}_t
=
\sum_{i=1}^{N}
g_t^{(i)}\hat{y}_t^{(i)},
\qquad
\sum_{i=1}^{N}g_t^{(i)}=1,
\qquad
g_t^{(i)}\ge 0,
```

where the gating weights $g_t^{(i)}$ depend on market volatility, model confidence, and recent model performance. This would let stable models dominate calm periods and reactive models receive more weight in stressed regimes.

## Useful Files

[`SQT-3.pdf`](SQT-3.pdf) is the written study document. It should be read together with the repository outputs: it explains the motivation, mathematical model choices, experimental design, result interpretation, and future-work direction.

[`data_non_std.csv`](data_non_std.csv) is the final modeling data set used by the training pipeline. It contains the S&P 500 return target and all explanatory features.

[`Selected_quant_tools.ipynb`](Selected_quant_tools.ipynb) constructs the data set. It gathers market prices, macro data, technical indicators, and trend features before producing the modeling table.

[`Base_models.py`](Base_models.py) defines the reusable base components: `AutoEncoder`, `ElasticNetLoss`, and `SharpeRatioLoss`. These are shared by the training pipeline and are mathematically central to the autoencoder and regularization experiments.

[`Simple_models.py`](Simple_models.py) defines the baseline neural architectures: FeedForward, LSTM, Transformer, CNN, and positional encoding.

[`TM_models.py`](TM_models.py) defines the more advanced temporal architectures: TCN and TFT, including causal dilated convolution blocks, residual temporal blocks, variable selection, static enrichment, LSTM processing, and attention.

[`environment.py`](environment.py) is the main experiment engine. It handles fold construction, standardization, optional autoencoder fitting, sequence construction, model training, evaluation metrics, plots, hyperparameter tuning, architecture selection, final training, and model comparison helpers.

[`Config_test.py`](Config_test.py) runs the two-stage selection logic: first architecture comparison under fixed settings, then hyperparameter optimization for the strongest architecture.

[`Arch_selection_final_training.py`](Arch_selection_final_training.py) trains selected architectures with and without autoencoder preprocessing and writes prediction outputs plus summary results.

[`final_model_training_autoencoder.py`](final_model_training_autoencoder.py) trains the final rolling-window autoencoder models and saves their fold models and prediction CSVs.

[`statistical analysis.py`](statistical%20analysis.py) computes the final forecast diagnostics: MAE, RMSE, $R^2$, Durbin-Watson, Jarque-Bera, Ljung-Box, and Diebold-Mariano p-value matrices.

[`saving.py`](saving.py) contains the persistence utilities for models and prediction files.

[`plotting.py`](plotting.py) creates real-vs-prediction figures from the saved prediction CSVs.

[`s.py`](s.py) extracts cleaned metric tables from serialized training summaries and writes `model_results.csv` files.

[`rolling/`](rolling) contains rolling-window results: trained outputs, saved autoencoder models, comparison metrics, Diebold-Mariano matrices, cleaned summaries, and final plots.

[`expanding/`](expanding) contains expanding-window results: trained outputs, comparison metrics, Diebold-Mariano matrices, cleaned summaries, and final plots.

