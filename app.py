import io, os, random, datetime as dt
from xml.sax.saxutils import escape
import streamlit as st, pandas as pd, numpy as np
from valusure.engine import ValuSure, HIGH, MEDIUM
from valusure.features import load_events
from valusure.geo import nearest
from valusure import rag

try:
    if os.environ.get("VALUSURE_NO_MAP"): raise ImportError
    import folium
    from streamlit_folium import st_folium
    HAS_MAP = True
except Exception:
    HAS_MAP = False

st.set_page_config(page_title="ValuSure prototype", layout="wide")
BASE_LTV = 0.75
OUTSIDE_KM = 4.0

@st.cache_resource(show_spinner="Training the model on the sample data...")
def load():
    df = pd.read_csv("data/sample_sales.csv")
    ev = load_events()
    eng = ValuSure(ev).fit(df[df.month_idx <= 31], df[df.month_idx > 31].iloc[:600])
    return eng, df, ev, pd.read_csv("data/localities.csv"), pd.read_csv("data/metro_stations.csv"), pd.read_csv("data/job_hubs.csv")

eng, df, events, localities, stations, hubs = load()
lakh = lambda v: f"Rs {v / 1e5:,.1f} lakh"
ss = st.session_state
ss.setdefault("lat", 18.5590); ss.setdefault("lon", 73.7870)
ss.setdefault("app_id", "VS-" + dt.date.today().strftime("%Y%m%d") + "-" + str(random.randint(1000, 9999)))

def credit_band(score):
    if score >= 750: return "Good (750 and above)", 0.75, "green"
    if score >= 700: return "Fair (700 to 749)", 0.70, "green"
    if score >= 650: return "Caution (650 to 699)", 0.65, "orange"
    return "Weak (below 650)", 0.0, "red"

st.title("ValuSure: confidence-routed property valuation")
st.caption("Prototype for the Hero Housing Fintech Challenge. Runs on SYNTHETIC sample data (Pune-style) and demo inputs. Nothing you type or upload is saved.")
t_file, t_val, t_news, t_sum = st.tabs(["1. Loan file", "2. Property and valuation", "3. News assistant (multilingual)", "4. Decision summary"])

# ------------------------------------------------------------------ 1. Loan file
with t_file:
    st.subheader("Loan application file")
    c1, c2, c3 = st.columns(3)
    lender = c1.text_input("Lender name", "Demo Lender Finance Ltd")
    c2.text_input("Application ID", ss.app_id, disabled=True)
    product = c3.selectbox("Product", ["Home Loan", "Loan Against Property"])
    borrower = c1.text_input("Borrower name", "Sample Borrower")
    id_type = c2.selectbox("ID proof type", ["Aadhaar", "PAN", "Voter ID", "Passport", "Driving licence"])
    id_last = c3.text_input("ID number: last 4 characters only", max_chars=4)
    id_file = st.file_uploader("ID proof scan (demo only)", type=["pdf", "png", "jpg", "jpeg"], key="idf")
    st.caption("Demo only: do not enter full ID numbers or upload real documents. Nothing is stored or verified here. In production, KYC would be verified through DigiLocker or CKYC with the borrower's consent.")

    st.markdown("**Credit and loan request**")
    d1, d2, d3 = st.columns(3)
    cibil = d1.slider("Credit score (CIBIL scale 300 to 900), demo input", 300, 900, 750)
    requested = d2.number_input("Loan amount requested (Rs lakh)", 5.0, 500.0, 70.0, 1.0)
    declared = d3.number_input("Declared or agreed property price (Rs lakh)", 5.0, 500.0, 105.0, 1.0)
    band_txt, ltv_cap, band_col = credit_band(cibil)
    st.markdown(f"Credit band: :{band_col}[{band_txt}]")
    st.caption("In production the score is fetched from a credit bureau with the borrower's consent. Score bands and LTV caps here are illustrative lender policy.")

    st.markdown("**Estimate and plan approval**")
    e1, e2 = st.columns(2)
    with e1:
        cost_file = st.file_uploader("Cost estimate (purchase or construction), demo", type=["pdf", "png", "jpg", "jpeg", "xlsx"], key="cost")
    with e2:
        plan_file = st.file_uploader("Sanctioned building plan (approval copy), demo", type=["pdf", "png", "jpg", "jpeg"], key="plan")
    p1, p2, p3, p4 = st.columns(4)
    plan_auth = p1.selectbox("Approving authority", ["PMC", "PCMC", "PMRDA", "Other"])
    plan_no = p2.text_input("Approval / permit number")
    plan_date = p3.date_input("Approval date", value=None)
    plan_status = p4.selectbox("Approval status", ["Approved", "Pending", "Not available"])
    st.caption("In production, plan approvals are verified against the municipal authority's portal. Here the status is self-declared.")

# ------------------------------------------------------------------ 2. Property and valuation
with t_val:
    st.subheader("Property location (GPS)")
    mcol, icol = st.columns([3, 2])
    with mcol:
        if HAS_MAP:
            m = folium.Map(location=[ss.lat, ss.lon], zoom_start=12, tiles="OpenStreetMap")
            for r in stations.itertuples():
                folium.CircleMarker([r.lat, r.lon], radius=4, color="purple", fill=True, tooltip=f"Metro: {r.name}").add_to(m)
            folium.Marker([ss.lat, ss.lon], tooltip="Property", icon=folium.Icon(color="red")).add_to(m)
            out = st_folium(m, height=380, use_container_width=True, returned_objects=["last_clicked"], key="map")
            clk = out.get("last_clicked") if out else None
            if clk and (clk["lat"], clk["lng"]) != ss.get("last_click"):
                ss.last_click = (clk["lat"], clk["lng"])
                ss.lat, ss.lon = round(clk["lat"], 5), round(clk["lng"], 5)
                st.rerun()
            st.caption("Click anywhere on the map to drop the property pin. Purple dots are sample metro stations (approximate positions).")
        else:
            st.map(pd.DataFrame({"lat": [ss.lat], "lon": [ss.lon]}))
            st.caption("Map click is unavailable (streamlit-folium not installed). Type the latitude and longitude instead.")
    with icol:
        st.number_input("Latitude", key="lat", format="%.5f", step=0.001)
        st.number_input("Longitude", key="lon", format="%.5f", step=0.001)
        bhk = st.selectbox("Bedrooms (BHK)", [1, 2, 3], index=1)
        area = st.number_input("Built-up area (sq ft)", 300, 3000, 950, 25)
        floor = st.slider("Floor", 0, 20, 5)
        age = st.slider("Building age (years)", 0, 40, 8)

    lat, lon = float(ss.lat), float(ss.lon)
    loc, d_loc = nearest(lat, lon, localities)
    stn_name, d_metro = nearest(lat, lon, stations)
    hub_name, d_hub = nearest(lat, lon, hubs)
    outside = d_loc > OUTSIDE_KM
    sub = df[df.locality == loc]
    green = st.slider("Green cover index (stands in for satellite imagery)", 0.0, 1.0, float(round(sub.green_index.median(), 2)), 0.01)

    g1, g2, g3 = st.columns(3)
    g1.metric("Locality (from GPS)", loc, f"{d_loc:.1f} km from locality centre", delta_color="off")
    g2.metric("Nearest metro station", stn_name, f"{d_metro:.1f} km", delta_color="off")
    g3.metric("Nearest job hub", hub_name, f"{d_hub:.1f} km", delta_color="off")
    if outside:
        st.warning(f"This pin is {d_loc:.1f} km from the nearest covered locality. It is outside the pilot area, so the case is routed to a physical valuer.")

    circle = float((sub.circle_rate_psf / 1.003 ** sub.month_idx).median() * 1.003 ** 36)
    row = dict(locality=loc, area_sqft=area, bhk=bhk, floor=floor, age_years=age, dist_metro_km=round(d_metro, 2),
               dist_hub_km=round(d_hub, 2), green_index=green, circle_rate_psf=circle, month_idx=36)
    res = eng.predict_row(row)
    route = "Physical valuer" if outside else res["route"]

    st.subheader("Valuation")
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated value", lakh(res["value"]))
    c2.metric("90% range", f"{res['low']/1e5:,.1f} to {res['high']/1e5:,.1f} lakh")
    c3.metric("Confidence", f"{res['confidence']:.0f} / 100")
    colour = {"Desktop valuation only": "green", "Tele-verification": "orange", "Physical valuer": "red"}[route]
    st.markdown(f"### Routing recommendation: :{colour}[{route}]")
    if res["speculative"]:
        st.info("Unverified or announced-only projects near this locality: the range was widened and the point value was not changed.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Main price drivers**")
        dr = pd.DataFrame(res["drivers"][:6], columns=["Factor", "Effect on value per sq ft (%)"]).set_index("Factor")
        st.bar_chart(dr, horizontal=True)
        st.caption("Bars to the right raise the value, bars to the left lower it, compared with an average property (approximate SHAP contributions).")
    with right:
        st.markdown("**Model agreement**")
        st.write(f"Gradient boosting model: **{lakh(res['lgb_value'])}**")
        st.write("Comparables engine: " + (f"**{lakh(res['comp_value'])}** from {res['n_comps']} similar sales" if res["n_comps"] and res["comp_value"] == res["comp_value"] else "too few similar sales"))
        st.write(f"Verified infrastructure score for this locality: **{res['infra_score']:.2f}**")
        st.markdown("**Most similar recent sales**")
        if res["n_comps"]:
            cm = res["comps"][["area_sqft", "bhk", "age_years", "month_idx", "price"]].copy()
            cm["price (lakh)"] = (cm.pop("price") / 1e5).round(1); cm["months ago"] = 36 - cm.pop("month_idx")
            st.dataframe(cm.rename(columns={"area_sqft": "area (sq ft)", "age_years": "age (yrs)"}), hide_index=True)
        else:
            st.write("No close comparables in the last 12 months.")
    with st.expander("Audit record"):
        st.json({"model": "ValuSure prototype v0.2", "time_trend_per_month": round(float(eng.trend), 5), "conformal_quantile": round(float(eng.q), 3),
                 "inputs": {**row, "lat": lat, "lon": lon}, "routing_thresholds": {"high": HIGH, "medium": MEDIUM}, "outside_pilot_area": outside})

# ------------------------------------------------------------------ 3. News assistant
with t_news:
    st.write(f"Ask about infrastructure near **{loc}** in English, Hindi or Tamil. Answers use only the verified sample events and cite the source.")
    lang_pick = st.radio("Sample questions in", list(rag.LANGS.values()), horizontal=True)
    code = [k for k, v in rag.LANGS.items() if v == lang_pick][0]
    q = st.selectbox("Pick a sample question", rag.SAMPLE_QUESTIONS[code])
    typed = st.text_input("...or type your own question")
    question = typed.strip() or q
    lang, ans, hits = rag.answer(question, loc, events)
    st.markdown(f"**Detected language:** {rag.LANGS[lang]}")
    st.success(ans)
    with st.expander("Sources used (sample, fictional events)"):
        st.dataframe(hits[["type", "name", "stage", "expected_year", "source", "source_tier", "published"]] if len(hits) else pd.DataFrame(), hide_index=True)
    st.caption("Prototype note: retrieval here is rule-based over a small sample table. The full design uses multilingual embeddings, a reranker and an LLM that answers only from retrieved sources.")

# ------------------------------------------------------------------ 4. Decision summary
with t_sum:
    st.subheader("Decision summary")
    ltv = min(BASE_LTV, ltv_cap)
    eligible = res["low"] * ltv
    dec_lakh = declared * 1e5
    dev = (dec_lakh - res["value"]) / res["value"] * 100
    price_flag = dec_lakh > res["high"]
    id_ok = id_file is not None and len(id_last) == 4
    cost_ok = cost_file is not None
    plan_ok = plan_status == "Approved" and bool(plan_no.strip())
    req_ok = requested * 1e5 <= eligible

    checks = [
        ("Borrower ID proof", "OK" if id_ok else "Missing", f"{id_type}, ending {id_last or '----'} (masked)"),
        ("Cost estimate", "OK" if cost_ok else "Missing", cost_file.name if cost_ok else "Not uploaded"),
        ("Plan approval", "OK" if plan_ok else ("Check" if plan_status == "Pending" else "Missing"), f"{plan_auth}, {plan_status}, no. {plan_no or 'not entered'}"),
        ("Credit score", {"green": "OK", "orange": "Check", "red": "Refer"}[band_col], f"{cibil}, {band_txt}"),
        ("Valuation routing", "OK" if route == "Desktop valuation only" else ("Check" if route == "Tele-verification" else "Refer"), route),
        ("Declared price vs model", "Refer" if price_flag else "OK", f"{dev:+.0f}% from model value, range {res['low']/1e5:,.1f} to {res['high']/1e5:,.1f} lakh"),
        ("Loan requested vs eligible", "OK" if req_ok else "Check", f"requested {requested:,.1f} lakh, eligible {eligible/1e5:,.1f} lakh at {int(ltv*100)}% LTV on the lower bound"),
    ]
    st.dataframe(pd.DataFrame(checks, columns=["Check", "Status", "Detail"]), hide_index=True)

    hard = price_flag or cibil < 650 or plan_status == "Not available" or outside
    missing = not (id_ok and cost_ok and plan_ok)
    if hard:
        rec, kind = "Refer to credit officer", "error"
    elif missing:
        rec, kind = "Pending: complete the missing documents", "warning"
    elif route == "Physical valuer":
        rec, kind = "Proceed after a physical valuation", "warning"
    elif not req_ok:
        rec, kind = f"Reduce the loan to the eligible limit of {eligible/1e5:,.1f} lakh", "warning"
    else:
        rec, kind = "Proceed to underwriting", "success"
    getattr(st, kind)(f"Recommended next step: {rec}")
    st.caption("Prototype logic for demonstration. The final credit decision always rests with the lender and its policy.")

    def build_pdf():
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        buf = io.BytesIO(); doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
        s = getSampleStyleSheet(); P = lambda t, st_="Normal": Paragraph(t, s[st_])
        def table(rows):
            t = Table([[P(escape(str(c))) for c in r] for r in rows], colWidths=[140, 375])
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")])); return t
        story = [P("ValuSure loan file summary (prototype, synthetic data)", "Title"),
                 P(f"Lender: {escape(lender)} | Application ID: {ss.app_id} | Product: {product} | Date: {dt.date.today()}"), Spacer(1, 8),
                 P("Borrower", "Heading3"), table([["Name", borrower], ["ID proof", f"{id_type}, ending {id_last or '----'} (masked)"], ["Credit score", f"{cibil}, {band_txt}"]]),
                 P("Property and valuation", "Heading3"),
                 table([["Location", f"{loc} ({lat:.5f}, {lon:.5f})"], ["Nearest metro", f"{stn_name}, {d_metro:.1f} km"],
                        ["Property", f"{bhk} BHK, {area} sq ft, floor {floor}, {age} years old"],
                        ["Estimated value", lakh(res["value"])], ["90% range", f"{res['low']/1e5:,.1f} to {res['high']/1e5:,.1f} lakh"],
                        ["Confidence", f"{res['confidence']:.0f} / 100"], ["Routing", route],
                        ["Main drivers", "; ".join(f"{n} {v:+.1f}%" for n, v in res["drivers"][:5])]]),
                 P("Lending view", "Heading3"),
                 table([["Declared price", f"{declared:,.1f} lakh ({dev:+.0f}% from model value)"], ["Loan requested", f"{requested:,.1f} lakh"],
                        ["Eligible loan (lower bound, LTV)", f"{eligible/1e5:,.1f} lakh at {int(ltv*100)}%"]]),
                 P("Checklist", "Heading3"), table([[a, f"{b}: {c}"] for a, b, c in checks]), Spacer(1, 8),
                 P(f"<b>Recommended next step: {escape(rec)}</b>"), Spacer(1, 6),
                 P("Prototype output on synthetic data and demo inputs. Not a valuation or credit decision.", "Italic")]
        doc.build(story); return buf.getvalue()
    st.download_button("Download loan file summary (PDF)", build_pdf(), file_name=f"{ss.app_id}_summary.pdf", mime="application/pdf")
