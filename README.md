# ValuSure: confidence-routed AI property valuation (prototype)

Prototype for the **Hero Housing Fintech Challenge, Problem Statement 1** (i-Hack, E-Summit 2026, IIT Bombay).

ValuSure returns, for a property: a **fair value**, a **90% range**, a **confidence score**, the **main price drivers**, the **most similar sales**, and a **routing recommendation** (desktop valuation only, tele-verification, or physical valuer). It also includes a map-based location step, a loan-file view, and a small **multilingual news assistant** (English, Hindi, Tamil).

> **Important:** this prototype runs on **synthetic** Pune-style sales data, **approximate** sample coordinates, and a **fictional** sample of infrastructure news. The numbers show that the pipeline works; they are not real-world accuracy. Replace the files in `data/` with real data to get real results. Nothing typed or uploaded in the app is stored.

## What the demo shows (four tabs)

1. **Loan file:** lender name, auto application ID, borrower name, ID proof (type, last 4 characters only, upload slot), credit score view with band, loan request, declared price, cost estimate upload, and plan approval details and upload.
2. **Property and valuation:** click the map to drop the property pin (or type latitude and longitude). Locality, nearest metro station and nearest job hub are computed from the GPS position using great-circle distance. Then value, 90% range, confidence, routing, SHAP price drivers, comparables and an audit record. A pin far outside the covered area is routed to a physical valuer.
3. **News assistant:** questions in English, Hindi or Tamil about metro, airport, industry, school and college projects near the pin, answered in the same language with sources.
4. **Decision summary:** checklist across ID, estimate, plan approval, credit band, valuation routing, declared-price check and loan eligibility (LTV on the lower bound, capped by credit band), a recommended next step, and a downloadable PDF of the loan file summary.

## How it maps to the design

| Design element | In this prototype |
|---|---|
| Time adjustment of old sales | City price trend fitted from data; sales and comparables adjusted to the valuation month |
| LightGBM point model | Predicts time-adjusted log price per sq ft, locality as a categorical feature |
| Comparables engine | Similar recent sales in the same locality; blended with LightGBM by data density |
| Uncertainty | Normalised split-conformal prediction gives an adaptive 90% range |
| Explainability | SHAP drivers, comparables table, audit record |
| Routing | Confidence score maps to desktop / tele-verification / physical valuer |
| GPS and geo features | Haversine distance from the pin to sample metro stations and job hubs |
| News layer | Events carry type, stage and source tier; only sanctioned-or-later credible events change the value; announced or rumoured items only widen the range |

**Simplified in the prototype (the full design is in the PDF):** nearest locality centre is used instead of H3 cells; a green-cover number stands in for satellite features (no CNN); news retrieval is rule-based instead of multilingual embeddings plus an LLM; the credit score, ID proof and plan approval are demo inputs that are not verified; metro and hub coordinates are approximate. Hindi and Tamil answer templates should be reviewed by a native speaker.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train_and_evaluate.py     # back-test vs baselines, writes results.md
streamlit run app.py             # opens the demo in your browser
```

`python data/make_data.py` regenerates the synthetic dataset (fixed random seed). If the map does not load, set `VALUSURE_NO_MAP=1` to use the latitude and longitude boxes instead.

## Back-test (synthetic data, out-of-time split: train months 0-27, calibrate 28-31, test 32-35)

|                                                |   MdAPE % |   MAPE % |   PPE10 % |   COD |   PRD |   90% interval coverage % |
|:-----------------------------------------------|----------:|---------:|----------:|------:|------:|--------------------------:|
| Baseline 1: circle rate                        |      7.96 |     8.74 |     62.89 |  7.78 |  1.01 |                    nan    |
| Baseline 2: hedonic regression                 |      4.92 |     6.04 |     83.38 |  5.99 |  1.01 |                    nan    |
| Baseline 3: locality-average AVM               |     11.67 |    11.49 |     41.12 |  7.5  |  1.01 |                    nan    |
| ValuSure (all cases)                           |      5.21 |     6.52 |     80.95 |  6.23 |  1.01 |                     92.26 |
| ValuSure (high-confidence only, 41% of cases)  |      4.59 |     5.25 |     89.4  |  4.97 |  1    |                     90.46 |
| Hedonic regression, same high-confidence cases |      4.14 |     4.83 |     91.52 |  4.78 |  1    |                    nan    |

**How to read this honestly.** On this synthetic data a simple hedonic regression is as accurate as ValuSure, and slightly better on the same high-confidence cases. That is expected: the data was generated from smooth relationships, which a regression fits well, and the sample is small and clean. What the prototype does show is (1) a **calibrated range**, with interval coverage close to the 90% target, and (2) a **confidence score that separates easier cases from harder ones**, which is what lets a lender route cases to desktop, tele-verification or a physical valuer. Whether the comparables, satellite and news features improve accuracy on real, messy, thinly traded markets must be tested on real transactions and lender valuation reports; this repository does not claim that result.

## Project layout

```
app.py                    Streamlit demo (loan file, valuation with map, news assistant, decision summary)
train_and_evaluate.py     Back-test and baseline comparison
valusure/engine.py        Model, comparables, conformal range, SHAP, routing
valusure/features.py      Feature list and news-event gating rules
valusure/geo.py           GPS distance helpers
valusure/rag.py           Multilingual news assistant (prototype)
data/make_data.py         Synthetic data generator
data/sample_sales.csv     Synthetic sales with coordinates (generated)
data/localities.csv       Approximate locality centres
data/metro_stations.csv   Approximate sample metro station positions
data/job_hubs.csv         Approximate job hub positions
data/infra_events.csv     Fictional sample infrastructure events
```

## Metrics

MdAPE (median absolute percentage error), PPE10 (share of valuations within 10% of the actual price), COD and PRD (IAAO ratio-study standards), and interval coverage.

## Licence and data

Code: MIT. All data in this repository is synthetic, fictional or approximate.
