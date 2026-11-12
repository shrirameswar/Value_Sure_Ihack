"""ValuSure prototype engine: time-adjusted LightGBM + comparables + adaptive conformal range + SHAP + routing."""
import numpy as np, pandas as pd, lightgbm as lgb, shap
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from .features import FEATURES, FRIENDLY, infra_scores

COVERAGE = 0.90
HIGH, MEDIUM = 56, 40   # routing thresholds, chosen so roughly a third of calibration cases route to desktop; a lender sets these by credit policy


class ValuSure:
    def __init__(self, events):
        self.events = events
        self.infra = infra_scores(events)

    # ---------- data prep
    def _prep(self, df):
        df = df.copy()
        df["locality_code"] = df["locality"].map(self.loc_codes).astype(int)
        df["infra_score"] = df["locality"].map(self.infra["infra_score"]).fillna(0.0)
        df["speculative"] = df["locality"].map(self.infra["speculative"]).fillna(0).astype(int)
        return df

    def fit(self, train, cal):
        self.loc_codes = {l: i for i, l in enumerate(sorted(train["locality"].unique()))}
        train, cal = self._prep(train), self._prep(cal)
        # 1) time adjustment: city price trend from log price per sq ft with locality effects
        X = pd.get_dummies(train["locality"]).astype(float); X["m"] = train["month_idx"]
        y = np.log(train["price"] / train["area_sqft"])
        self.trend = LinearRegression().fit(X, y).coef_[-1]
        self.m_ref = 0
        # 2) point model on time-adjusted log price per sq ft
        self.known = pd.concat([train, cal], ignore_index=True)
        adj = lambda d: np.log(d["price"] / d["area_sqft"]) - self.trend * (d["month_idx"] - self.m_ref)
        params = dict(n_estimators=500, learning_rate=0.04, num_leaves=31, min_child_samples=20,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1, random_state=1)
        self.model = lgb.LGBMRegressor(**params).fit(train[FEATURES], adj(train), categorical_feature=["locality_code"])
        # 3) uncertainty: model of expected error, then conformal calibration on held-out data
        oof = np.zeros(len(train))
        for tr, te in KFold(5, shuffle=True, random_state=0).split(train):
            m = lgb.LGBMRegressor(**{**params, "n_estimators": 250}).fit(train.iloc[tr][FEATURES], adj(train).iloc[tr],
                                                                       categorical_feature=["locality_code"])
            oof[te] = adj(train).iloc[te] - m.predict(train.iloc[te][FEATURES])
        self.sigma_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=1)
        self.sigma_model.fit(train[FEATURES], np.abs(oof), categorical_feature=["locality_code"])
        sig = np.clip(self.sigma_model.predict(cal[FEATURES]), 0.02, None)
        score = np.abs(adj(cal) - self.model.predict(cal[FEATURES])) / sig
        n = len(score); self.q = np.quantile(score, min(1, np.ceil((n + 1) * COVERAGE) / n))
        self.explainer = shap.TreeExplainer(self.model)
        return self

    # ---------- comparables
    def comparables(self, row, k=5):
        d = self.known
        c = d[(d["locality"] == row["locality"]) & (d["bhk"] == row["bhk"]) &
              (d["area_sqft"].between(row["area_sqft"] * .8, row["area_sqft"] * 1.2)) &
              (d["month_idx"] <= row["month_idx"]) & (d["month_idx"] >= row["month_idx"] - 12)].copy()
        if c.empty:
            return c, np.nan
        c["adj_psf"] = c["price"] / c["area_sqft"] * np.exp(self.trend * (row["month_idx"] - c["month_idx"]))
        c["similarity"] = 1 / (1 + abs(c["area_sqft"] - row["area_sqft"]) / 200 + abs(c["age_years"] - row["age_years"]) / 10
                               + (row["month_idx"] - c["month_idx"]) / 6)
        c = c.sort_values("similarity", ascending=False)
        top = c.head(max(k, 5))
        psf = np.average(top["adj_psf"], weights=top["similarity"]) if len(c) >= 3 else np.nan
        return c.head(k), psf * row["area_sqft"] if psf == psf else np.nan

    # ---------- valuation
    def predict_row(self, row):
        r = self._prep(pd.DataFrame([row])).iloc[0]
        X = pd.DataFrame([r[FEATURES]]).astype(float)
        base = self.model.predict(X)[0] + self.trend * (r["month_idx"] - self.m_ref)
        lgb_val = np.exp(base) * r["area_sqft"]
        comps, comp_val = self.comparables(r.to_dict())
        n_comps = len(comps)
        w = 0.0 if comp_val != comp_val else min(0.4, 0.04 * n_comps)
        point = lgb_val if w == 0 else (1 - w) * lgb_val + w * comp_val
        sig = max(float(self.sigma_model.predict(X)[0]), 0.02)
        half = self.q * sig * (1.10 if r["speculative"] else 1.0)   # unverified news widens the range
        lo, hi = point * np.exp(-half), point * np.exp(half)
        agree = 1.0 if w == 0 else 1 - min(abs(lgb_val - comp_val) / lgb_val, 0.2) / 0.2 * 0.3
        conf = 100 * np.clip(1 - half / 0.30, 0, 1) * agree * (0.85 if n_comps < 3 else 1.0)
        route = "Desktop valuation only" if conf >= HIGH else ("Tele-verification" if conf >= MEDIUM else "Physical valuer")
        sv = self.explainer.shap_values(X)[0]
        drivers = sorted(zip([FRIENDLY[f] for f in FEATURES], sv * 100), key=lambda t: -abs(t[1]))
        return dict(value=point, low=lo, high=hi, confidence=float(conf), route=route, lgb_value=lgb_val,
                    comp_value=comp_val, n_comps=n_comps, comps=comps, drivers=drivers,
                    speculative=bool(r["speculative"]), infra_score=float(r["infra_score"]))

    def predict_df(self, df):
        out = []
        for _, r in df.iterrows():
            p = self.predict_row(r.to_dict())
            out.append((p["value"], p["low"], p["high"], p["confidence"], p["route"]))
        return pd.DataFrame(out, columns=["value", "low", "high", "confidence", "route"], index=df.index)
