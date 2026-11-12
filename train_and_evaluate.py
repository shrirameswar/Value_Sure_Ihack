"""Back-test ValuSure against 3 baselines with an out-of-time split. Run: python train_and_evaluate.py"""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from valusure.engine import ValuSure, HIGH
from valusure.features import load_events, FEATURES

df = pd.read_csv("data/sample_sales.csv")
train, cal, test = df[df.month_idx <= 27], df[(df.month_idx > 27) & (df.month_idx <= 31)], df[df.month_idx > 31]
print(f"train {len(train)} | calibration {len(cal)} | test (latest 4 months) {len(test)}")

eng = ValuSure(load_events()).fit(train, cal)
res = eng.predict_df(test); actual = test["price"]

def metrics(pred, act, lo=None, hi=None):
    ape = (pred - act).abs() / act; ratio = pred / act
    out = {"MdAPE %": 100 * ape.median(), "MAPE %": 100 * ape.mean(), "PPE10 %": 100 * (ape <= .10).mean(),
           "COD": 100 * (ratio - ratio.median()).abs().mean() / ratio.median(), "PRD": ratio.mean() / (pred.sum() / act.sum())}
    if lo is not None: out["90% interval coverage %"] = 100 * ((act >= lo) & (act <= hi)).mean()
    return out

rows = {}
# Baseline 1: circle rate scaled by median training ratio
k = (train.price / train.area_sqft / train.circle_rate_psf).median()
rows["Baseline 1: circle rate"] = metrics(test.circle_rate_psf * k * test.area_sqft, actual)
# Baseline 2: simple hedonic regression (no satellite/infra, locality dummies, time adjusted)
Xtr = pd.get_dummies(train[["locality"]]).astype(float).join(train[["area_sqft", "bhk", "floor", "age_years", "month_idx"]])
Xte = pd.get_dummies(test[["locality"]]).astype(float).join(test[["area_sqft", "bhk", "floor", "age_years", "month_idx"]]).reindex(columns=Xtr.columns, fill_value=0)
h = Ridge(alpha=1.0).fit(Xtr, np.log(train.price / train.area_sqft))
hed = pd.Series(np.exp(h.predict(Xte)) * test.area_sqft.values, index=test.index)
rows["Baseline 2: hedonic regression"] = metrics(hed, actual)
# Baseline 3: basic AVM = locality median price per sq ft (no time adjustment)
med = (train.price / train.area_sqft).groupby(train.locality).median()
rows["Baseline 3: locality-average AVM"] = metrics(test.locality.map(med) * test.area_sqft, actual)
rows["ValuSure (all cases)"] = metrics(res.value, actual, res.low, res.high)
hi = res.confidence >= HIGH
rows[f"ValuSure (high-confidence only, {100*hi.mean():.0f}% of cases)"] = metrics(res.value[hi], actual[hi], res.low[hi], res.high[hi])

rows["Hedonic regression, same high-confidence cases"] = metrics(hed[hi], actual[hi])
tab = pd.DataFrame(rows).T.round(2)
print(tab.to_string())
print("\nRouting mix:", (res.route.value_counts(normalize=True) * 100).round(1).to_dict())
with open("results.md", "w") as f:
    f.write("# Back-test results (SYNTHETIC data, out-of-time split)\n\n" + tab.to_markdown() + "\n\n")
    f.write("Routing mix (% of test cases): " + str((res.route.value_counts(normalize=True) * 100).round(1).to_dict()) + "\n")
