"""Prototype news assistant: language detection, intent detection, retrieval over verified events, answers in the user's language.
PROTOTYPE NOTE: retrieval here is rule-based over a small sample table. The production design uses multilingual
embeddings + a reranker + an LLM that answers only from retrieved context (see the design document, Section 10)."""
import re
import pandas as pd

LANGS = {"en": "English", "hi": "Hindi", "ta": "Tamil"}
INTENT_WORDS = {
    "metro": ["metro", "मेट्रो", "மெட்ரோ"],
    "airport": ["airport", "flight", "हवाई", "एयरपोर्ट", "विमान", "விமான"],
    "industry": ["industry", "industrial", "it park", "tech park", "job", "factory", "उद्योग", "आईटी", "नौकरी", "தொழில்", "ஐடி", "வேலை"],
    "school": ["school", "स्कूल", "பள்ளி"],
    "college": ["college", "university", "campus", "कॉलेज", "विश्वविद्यालय", "கல்லூரி", "பல்கலை"],
}
T = {
 "en": {"type": {"metro": "metro", "airport": "airport", "industry": "industry / IT park", "school": "school", "college": "college"},
        "stage": {"rumoured": "unverified rumour", "announced": "announced", "sanctioned": "sanctioned", "under_construction": "under construction", "operational": "operational"},
        "line": "Near {loc}: {type}, {name}. Stage: {stage}. Expected year: {year}. Source: {src}.",
        "none": "No verified information was found for this locality.",
        "caution": "This is indicative information, not a guarantee of any price change. Confirm with the official source."},
 "hi": {"type": {"metro": "मेट्रो", "airport": "हवाई अड्डा", "industry": "उद्योग / आईटी पार्क", "school": "स्कूल", "college": "कॉलेज"},
        "stage": {"rumoured": "असत्यापित अफवाह", "announced": "घोषित", "sanctioned": "स्वीकृत", "under_construction": "निर्माणाधीन", "operational": "चालू"},
        "line": "{loc} के पास {type}: {name}। स्थिति: {stage}। अनुमानित वर्ष: {year}। स्रोत: {src}।",
        "none": "इस इलाके के लिए कोई सत्यापित जानकारी नहीं मिली।",
        "caution": "यह केवल सूचक जानकारी है, कीमत बढ़ने की कोई गारंटी नहीं। आधिकारिक स्रोत से पुष्टि करें।"},
 "ta": {"type": {"metro": "மெட்ரோ", "airport": "விமான நிலையம்", "industry": "தொழில் / ஐடி பூங்கா", "school": "பள்ளி", "college": "கல்லூரி"},
        "stage": {"rumoured": "சரிபார்க்கப்படாத வதந்தி", "announced": "அறிவிக்கப்பட்டது", "sanctioned": "அனுமதிக்கப்பட்டது", "under_construction": "கட்டுமானத்தில்", "operational": "செயல்பாட்டில்"},
        "line": "{loc} அருகில் {type}: {name}. நிலை: {stage}. எதிர்பார்க்கப்படும் ஆண்டு: {year}. ஆதாரம்: {src}.",
        "none": "இந்தப் பகுதிக்கு சரிபார்க்கப்பட்ட தகவல் கிடைக்கவில்லை.",
        "caution": "இது குறிப்பிற்கான தகவல் மட்டுமே; விலை உயர்வுக்கு உத்தரவாதம் இல்லை. அதிகாரப்பூர்வ ஆதாரத்தில் உறுதிசெய்யவும்."},
}
SAMPLE_QUESTIONS = {
    "en": ["Is any metro coming near this property?", "Are there colleges or schools nearby?", "Any new airport or industry projects?"],
    "hi": ["क्या इस संपत्ति के पास कोई मेट्रो आ रही है?", "क्या आसपास कॉलेज या स्कूल हैं?", "क्या कोई नया उद्योग या हवाई अड्डा परियोजना है?"],
    "ta": ["இந்தச் சொத்தின் அருகில் மெட்ரோ வருகிறதா?", "அருகில் கல்லூரி அல்லது பள்ளி உள்ளதா?", "புதிய விமான நிலையம் அல்லது தொழில் திட்டங்கள் உள்ளதா?"],
}


def detect_language(text):
    if re.search(r"[\u0900-\u097F]", text): return "hi"
    if re.search(r"[\u0B80-\u0BFF]", text): return "ta"
    return "en"


def detect_intents(text):
    t = text.lower()
    found = [k for k, words in INTENT_WORDS.items() if any(w in t for w in words)]
    return found or list(INTENT_WORDS)   # no specific topic: show everything nearby


def answer(question, locality, events: pd.DataFrame):
    lang = detect_language(question); tr = T[lang]
    intents = detect_intents(question)
    hits = events[(events["locality"] == locality) & (events["type"].isin(intents))]
    hits = hits[hits["source_tier"] != "social"].copy() if len(hits[hits["source_tier"] != "social"]) else hits
    if hits.empty:
        return lang, tr["none"] + "\n\n" + tr["caution"], hits
    lines = [tr["line"].format(loc=locality, type=tr["type"][e["type"]], name=e["name"], stage=tr["stage"][e["stage"]],
                               year=e["expected_year"], src=e["source"]) for _, e in hits.iterrows()]
    return lang, "\n\n".join(lines) + "\n\n" + tr["caution"], hits
