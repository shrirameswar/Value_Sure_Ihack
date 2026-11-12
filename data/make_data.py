"""Generate a SYNTHETIC Pune-style sales dataset for the prototype (with GPS coordinates).
This is NOT real transaction data. Replace data/sample_sales.csv with real data to use the engine for real."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from valusure.features import load_events, infra_scores
from valusure.geo import haversine_km

rng = np.random.default_rng(42)
LOC = {  # base price per sq ft, green index, noise level
 "Baner": (9500, .35, .05), "Hinjewadi": (7500, .30, .06), "Kothrud": (10500, .30, .05), "Wakad": (8000, .32, .05),
 "Hadapsar": (7200, .20, .06), "Viman Nagar": (10000, .25, .05), "Kharadi": (8800, .28, .06), "Aundh": (10200, .38, .05),
 "Pimple Saudagar": (7800, .27, .06), "Wagholi": (5200, .40, .09), "Undri": (6500, .42, .09), "Bavdhan": (9000, .45, .06)}
locs = pd.read_csv("data/localities.csv").set_index("name")
stn = pd.read_csv("data/metro_stations.csv"); hubs = pd.read_csv("data/job_hubs.csv")
ev = load_events(); inf = infra_scores(ev)
nearest_d = lambda lat, lon, df: float(haversine_km(lat, lon, df.lat.to_numpy(), df.lon.to_numpy()).min())
rows = []
for i in range(6000):
    loc = rng.choice(list(LOC)); base, gr, nz = LOC[loc]
    lat = locs.loc[loc, "lat"] + rng.normal(0, .009); lon = locs.loc[loc, "lon"] + rng.normal(0, .009)
    d_metro = nearest_d(lat, lon, stn); d_hub = nearest_d(lat, lon, hubs)
    bhk = rng.choice([1, 2, 3], p=[.2, .5, .3])
    area = {1: rng.uniform(450, 650), 2: rng.uniform(750, 1100), 3: rng.uniform(1100, 1600)}[bhk]
    age = int(rng.integers(0, 30)); floor = int(rng.integers(0, 20)); m = int(rng.integers(0, 36))
    green = np.clip(gr + rng.normal(0, .06), 0, 1)
    infra = inf.loc[loc, "infra_score"] if loc in inf.index else 0.0
    psf = (base * (1 + .05 * infra) * np.exp(-.006 * age) * (1 + .004 * min(floor, 15)) * (1 - .012 * min(d_metro, 12))
           * (1 - .008 * min(d_hub, 12)) * (1 + .12 * green) * (1.006 ** m))
    psf *= np.exp(rng.normal(0, nz + (.03 if age > 20 else 0)))
    rows.append(dict(locality=loc, lat=round(lat, 5), lon=round(lon, 5), area_sqft=round(area), bhk=bhk, floor=floor, age_years=age,
                     dist_metro_km=round(d_metro, 2), dist_hub_km=round(d_hub, 2), green_index=round(float(green), 3),
                     circle_rate_psf=round(base * .55 * (1.003 ** m)), month_idx=m, price=round(psf * area, -3)))
pd.DataFrame(rows).to_csv("data/sample_sales.csv", index=False)
print("wrote data/sample_sales.csv", len(rows), "rows (synthetic)")
