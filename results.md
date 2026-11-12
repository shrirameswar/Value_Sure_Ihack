# Back-test results (SYNTHETIC data, out-of-time split)

|                                                |   MdAPE % |   MAPE % |   PPE10 % |   COD |   PRD |   90% interval coverage % |
|:-----------------------------------------------|----------:|---------:|----------:|------:|------:|--------------------------:|
| Baseline 1: circle rate                        |      7.96 |     8.74 |     62.89 |  7.78 |  1.01 |                    nan    |
| Baseline 2: hedonic regression                 |      4.92 |     6.04 |     83.38 |  5.99 |  1.01 |                    nan    |
| Baseline 3: locality-average AVM               |     11.67 |    11.49 |     41.12 |  7.5  |  1.01 |                    nan    |
| ValuSure (all cases)                           |      5.21 |     6.52 |     80.95 |  6.23 |  1.01 |                     92.26 |
| ValuSure (high-confidence only, 41% of cases)  |      4.59 |     5.25 |     89.4  |  4.97 |  1    |                     90.46 |
| Hedonic regression, same high-confidence cases |      4.14 |     4.83 |     91.52 |  4.78 |  1    |                    nan    |

Routing mix (% of test cases): {'Desktop valuation only': 40.5, 'Tele-verification': 38.7, 'Physical valuer': 20.8}
