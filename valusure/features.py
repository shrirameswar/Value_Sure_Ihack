import pandas as pd

TYPE_WEIGHT = {"metro": 1.0, "airport": 0.8, "industry": 0.7, "college": 0.4, "school": 0.3}
STAGE_WEIGHT = {"rumoured": 0.0, "announced": 0.1, "sanctioned": 0.4, "under_construction": 0.7, "operational": 1.0}
TIER_WEIGHT = {"gov": 1.0, "national": 0.8, "local": 0.5, "social": 0.0}
# Gating rule: only events at sanctioned stage or later, from a credible source, move the point value.
VALUE_STAGES = {"sanctioned", "under_construction", "operational"}
SPECULATIVE_STAGES = {"announced", "rumoured"}


def load_events(path="data/infra_events.csv"):
    return pd.read_csv(path)


def infra_scores(events: pd.DataFrame) -> pd.DataFrame:
    """Per-locality infrastructure score (feeds the model) and a speculative flag (widens the range)."""
    rows = {}
    for loc, g in events.groupby("locality"):
        val = 0.0
        for _, e in g.iterrows():
            if e["stage"] in VALUE_STAGES:
                val += TYPE_WEIGHT[e["type"]] * STAGE_WEIGHT[e["stage"]] * TIER_WEIGHT[e["source_tier"]]
        spec = any(s in SPECULATIVE_STAGES for s in g["stage"])
        rows[loc] = {"infra_score": min(val, 2.0), "speculative": int(spec)}
    return pd.DataFrame.from_dict(rows, orient="index")


FEATURES = ["area_sqft", "bhk", "floor", "age_years", "dist_metro_km", "dist_hub_km",
            "green_index", "circle_rate_psf", "infra_score", "locality_code"]
FRIENDLY = {"area_sqft": "Built-up area", "bhk": "Number of bedrooms", "floor": "Floor",
            "age_years": "Building age", "dist_metro_km": "Distance to metro",
            "dist_hub_km": "Distance to job hub", "green_index": "Green cover (satellite proxy)",
            "circle_rate_psf": "Government circle rate", "infra_score": "Verified infrastructure projects",
            "locality_code": "Locality"}
