"""
LifeLine — Premium Kaggle Notebook (Python script)
Convert to .ipynb via VS Code or run cells in order.

PURPOSE:
- Production-style ADK-first multi-agent pipeline for crisis detection, verification, geocoding,
  triage/priority, A2A orchestration, observability, memory, and report generation.
- Real-API mode: uses Google Geocoding API (requires GOOGLE_API_KEY).

USAGE:
1) Put this file into your LifeLine folder (created earlier).
2) Set GOOGLE_API_KEY in your environment (or Kaggle Secrets).
   - Windows PowerShell: $env:GOOGLE_API_KEY="YOUR_KEY"
   - Linux / macOS: export GOOGLE_API_KEY="YOUR_KEY"
3) From VS Code: open notebook.py -> Command Palette -> "Jupyter: Create Notebook from Python File".
4) Run cells in notebook interactively and follow the demo sections.

NOTE:
- Replace placeholder ADK Agent registration with your course ADK client imports if required.
- This script uses a TF-IDF baseline detector. Replace with fine-tuned models (DistilBERT etc.) for final evaluation.
"""

# -----------------------------------------------------------------------------
# CELL 1 — Install / environment reminder (run once in a new kernel)
# -----------------------------------------------------------------------------
# (Uncomment and run these lines in an environment where you can install packages)
# !pip install --quiet pandas numpy scikit-learn spacy geopy sqlalchemy streamlit pydeck reportlab sentence-transformers faiss-cpu google-cloud-vision
# !python -m spacy download en_core_web_sm

# -----------------------------------------------------------------------------
# CELL 2 — Imports & Logging
# -----------------------------------------------------------------------------
import os
import sys
import time
import json
import logging
import math
from datetime import datetime, timedelta
from pprint import pprint
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np

# Geocoding
from geopy.geocoders import GoogleV3

# Text processing & baseline ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# NER
import spacy

# Persistence
from sqlalchemy import create_engine, text

# Observability extras
import uuid
import traceback

try:
    from google.cloud import vision
    GCV_AVAILABLE = True
except Exception:
    GCV_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("lifeline")

# -----------------------------------------------------------------------------
# CELL 3 — Configuration & API Keys (REAL API MODE)
# -----------------------------------------------------------------------------
# You must set GOOGLE_API_KEY before running this notebook. In Kaggle use Secrets.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Please set GOOGLE_API_KEY in environment before running (use Kaggle Secrets).")

# SQLite memory DB (persistent across notebook runs)
DB_FILE = "lifeline_memory.db"
SQLITE_URL = f"sqlite:///{DB_FILE}"
engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})

# Ensure reports table exists
with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,
            text TEXT,
            ts TEXT,
            lat REAL,
            lon REAL,
            type TEXT,
            prob REAL,
            urgency REAL,
            verification_score REAL
        )
    """))

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# -----------------------------------------------------------------------------
# CELL 4 — Utility functions
# -----------------------------------------------------------------------------
def now_iso():
    return datetime.utcnow().isoformat()

def haversine_km(lat1, lon1, lat2, lon2):
    # returns distance in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return 6371.0 * c

def safe_geo_to_dict(geo):
    if not geo:
        return {"lat": None, "lon": None, "address": None}
    return {"lat": geo.get("lat"), "lon": geo.get("lon"), "address": geo.get("address")}

# -----------------------------------------------------------------------------
# CELL 5 — MCP-style Tools (Geocode, Weather, Memory, Image Verification)
# -----------------------------------------------------------------------------
class GeocodeTool:
    """MCP-style wrapper for Google Geocoding (via geopy.GoogleV3)."""
    def __init__(self, api_key: str):
        self.geolocator = GoogleV3(api_key=api_key, timeout=10)

    def call(self, query: str) -> Optional[Dict[str, Any]]:
        """Return {'lat', 'lon', 'address'} or None."""
        if not query or not str(query).strip():
            return None
        try:
            res = self.geolocator.geocode(query)
            if res:
                return {"lat": res.latitude, "lon": res.longitude, "address": res.address}
        except Exception as e:
            logger.exception("GeocodeTool error")
        return None

class WeatherTool:
    """Placeholder weather tool (swap in OpenWeather / other API)."""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def call(self, lat: float, lon: float) -> Dict[str, Any]:
        # Replace with a call to OpenWeather or similar if desired.
        # Here we return a synthetic safe response for demo:
        return {"temp_c": 28.0, "precip_prob": 0.1}

class MemoryTool:
    """Simple SQL-backed Memory Tool (persistence)."""
    def __init__(self, engine):
        self.engine = engine

    def call(self, action: str, payload: Optional[dict] = None):
        if action == "insert_report" and payload:
            df = pd.DataFrame([payload])
            df.to_sql("reports", self.engine, if_exists="append", index=False)
            return True
        if action == "query_recent":
            since = payload.get("since")
            q = "SELECT * FROM reports WHERE ts >= :since"
            with self.engine.connect() as conn:
                df = pd.read_sql_query(text(q), conn, params={"since": since})
                return df
        if action == "all_reports":
            with self.engine.connect() as conn:
                df = pd.read_sql_table("reports", conn)
                return df
        return None

class ImageVerificationTool:
    """
    Image verification using Google Cloud Vision (optional).
    If GCV credentials are not configured, this will return a mocked low-confidence result.
    """
    def __init__(self):
        self.available = GCV_AVAILABLE
        if self.available:
            try:
                self.client = vision.ImageAnnotatorClient()
            except Exception:
                self.available = False
                logger.warning("Google Cloud Vision client not available; falling back to mock verification.")

    def call(self, image_bytes: bytes) -> Dict[str, Any]:
        if self.available:
            try:
                image = vision.Image(content=image_bytes)
                # Example: label detection + web detection
                web = self.client.web_detection(image=image)
                best_guess = getattr(web, "best_guess_labels", None)
                # Use web detection results to estimate provenance/duplication
                # For demo, we compute a simple score based on web matches
                score = 0.0
                if web and getattr(web, "pages_with_matching_images", None):
                    score = min(1.0, 0.5 + 0.1 * len(web.pages_with_matching_images))
                return {"match_score": score, "provenance": "web", "details": str(web)}
            except Exception:
                logger.exception("Image verification via GCV failed")
                return {"match_score": 0.0, "provenance": "unknown", "details": None}
        # Mocked fallback
        return {"match_score": 0.0, "provenance": "mock", "details": None}

# Instantiate tools
geotool = GeocodeTool(GOOGLE_API_KEY)
weathertool = WeatherTool()
memory_tool = MemoryTool(engine)
img_tool = ImageVerificationTool()

# -----------------------------------------------------------------------------
# CELL 6 — Agent Implementations (Sequential + Parallel patterns)
# Note: These are ADK-style agent bodies. Replace AgentBase with course-supplied ADK classes if required.
# -----------------------------------------------------------------------------
class SignalDetectionAgent:
    """Baseline detection (TF-IDF + Logistic Regression). Replace with LLM/transformer for final runs."""
    def __init__(self):
        # seed simple baseline; in final submission swap with fine-tuned model on TREC-IS/HumAID
        texts = [
            "fire in building need help", "flooding near river", "what a nice day", "random tweet",
            "earthquake tremor heard", "car accident on highway"
        ]
        labels = [1,1,0,0,1,1]
        self.pipe = make_pipeline(TfidfVectorizer(ngram_range=(1,2), max_features=5000), LogisticRegression(max_iter=1000))
        self.pipe.fit(texts, labels)

    def act(self, message: dict) -> Dict[str, Any]:
        text = message.get("text", "")
        prob = float(self.pipe.predict_proba([text])[0][1])
        # simple type heuristics
        low = text.lower()
        dtype = "other"
        if any(k in low for k in ["flood","water","submerged"]): dtype = "flood"
        if any(k in low for k in ["fire","smoke","ablaze"]): dtype = "fire"
        if any(k in low for k in ["earthquake","tremor","quake"]): dtype = "earthquake"
        if any(k in low for k in ["accident","crash","collision"]): dtype = "accident"
        return {"is_disaster": prob >= 0.5, "prob": prob, "type": dtype}

class GeoExtractionAgent:
    """NER-based location extraction and geocoding tool-calling."""
    def __init__(self, geotool: GeocodeTool):
        self.geotool = geotool

    def act(self, message: dict) -> Optional[Dict[str, Any]]:
        text = message.get("text", "")
        doc = nlp(text)
        loc_entities = [ent.text for ent in doc.ents if ent.label_ in ("GPE","LOC","FAC","ORG")]
        # Try entities first
        for ent in loc_entities:
            try:
                geo = self.geotool.call(ent)
                if geo:
                    return {"lat": geo["lat"], "lon": geo["lon"], "address": geo["address"], "place": ent}
            except Exception:
                continue
        # fallback: geocode the full text (rarely works but possible)
        geo = self.geotool.call(text)
        if geo:
            return {"lat": geo["lat"], "lon": geo["lon"], "address": geo["address"], "place": None}
        return None

class VerificationAgent:
    """Verifies evidence (images or cross-source corroboration)."""
    def __init__(self, img_tool: ImageVerificationTool):
        self.img_tool = img_tool

    def act(self, message: dict) -> Dict[str, Any]:
        # If image present, verify
        if message.get("image_bytes"):
            res = self.img_tool.call(message["image_bytes"])
            return {"image_verification": res, "cross_source": []}
        # Placeholder: cross-source corroboration (search, other feeds) — mock low-cost check
        # In production use MCP search tools / Google Search API etc.
        return {"image_verification": None, "cross_source": []}

class PriorityAssessmentAgent:
    """Computes urgency using detection prob, keywords, weather, and memory frequency."""
    def __init__(self, weather_tool: WeatherTool, memory_tool: MemoryTool):
        self.weather_tool = weather_tool
        self.memory_tool = memory_tool
        self.keywords = ["help","trapped","urgent","immediately","now","rescue","send help"]

    def act(self, message: dict, geo: Optional[dict], detection: dict) -> dict:
        base = detection.get("prob", 0.0)
        urgency = base
        text = message.get("text", "").lower()
        for k in self.keywords:
            if k in text:
                urgency += 0.12
        # Weather influence
        if geo:
            w = self.weather_tool.call(geo["lat"], geo["lon"])
            if w.get("precip_prob", 0) > 0.5:
                urgency += 0.08
        # Historical frequency near location
        freq = 0
        if geo:
            since = (datetime.utcnow() - timedelta(days=30)).isoformat()
            hist = self.memory_tool.call("query_recent", {"since": since})
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                close = 0
                for _, r in hist.iterrows():
                    if pd.isna(r["lat"]) or pd.isna(r["lon"]):
                        continue
                    try:
                        dist = haversine_km(float(r["lat"]), float(r["lon"]), float(geo["lat"]), float(geo["lon"]))
                        if dist <= 5.0:
                            close += 1
                    except Exception:
                        continue
                freq = close
                urgency += min(0.2, 0.02 * freq)
        urgency = min(1.0, float(urgency))
        return {"urgency": urgency, "historical_frequency": freq}

class ReporterAgent:
    """Generates JSON and simple textual reports; can be extended to PDF export."""
    def __init__(self, out_dir: str = "reports"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def act(self, packet: dict) -> str:
        event_id = packet.get("event_id", f"E-{uuid.uuid4().hex}")
        packet["event_id"] = event_id
        path = os.path.join(self.out_dir, f"{event_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(packet, f, indent=2)
        # Also return a short human-readable report (string)
        report_text = (
            f"Event: {event_id}\nType: {packet.get('type')}\n"
            f"Coords: {packet.get('coordinates')}\nUrgency: {packet.get('urgency')}\nGenerated: {packet.get('generated_at')}"
        )
        with open(os.path.join(self.out_dir, f"{event_id}.txt"), "w", encoding="utf-8") as f:
            f.write(report_text)
        return path

# -----------------------------------------------------------------------------
# CELL 7 — Coordinator Agent (A2A orchestration)
# -----------------------------------------------------------------------------
class CoordinatorAgent:
    """
    Orchestrates A2A messages between agents and returns a final dispatch packet.
    Demonstrates sequenced A2A call pattern.
    """
    def __init__(self, detector: SignalDetectionAgent, verifier: VerificationAgent,
                 geo_agent: GeoExtractionAgent, priority_agent: PriorityAssessmentAgent,
                 reporter: ReporterAgent, memory_tool: MemoryTool):
        self.detector = detector
        self.verifier = verifier
        self.geo_agent = geo_agent
        self.priority_agent = priority_agent
        self.reporter = reporter
        self.memory_tool = memory_tool

    def handle_message(self, message: dict) -> dict:
        trace = []
        # Step 1: Detection
        det = self.detector.act(message)
        trace.append({"step": "detection", "output": det, "ts": now_iso()})
        if not det.get("is_disaster", False):
            return {"status": "not_disaster", "trace": trace}

        # Step 2: Verification
        ver = self.verifier.act(message)
        trace.append({"step": "verification", "output": ver, "ts": now_iso()})
        # If verification is very low, we might mark as needs human review (policy)
        if ver.get("image_verification") and ver["image_verification"].get("match_score", 0) < 0.1:
            # Continue but add low-confidence flag
            human_review = True
        else:
            human_review = False

        # Step 3: Geo Extraction
        geo = self.geo_agent.act(message)
        trace.append({"step": "geo", "output": safe_geo_to_dict(geo), "ts": now_iso()})

        # Step 4: Priority / Triage
        pri = self.priority_agent.act(message, geo, det)
        trace.append({"step": "triage", "output": pri, "ts": now_iso()})

        # Build dispatch packet
        packet = {
            "event_id": f"E-{int(time.time()*1000)}",
            "type": det.get("type"),
            "coordinates": [geo.get("lat") if geo else None, geo.get("lon") if geo else None],
            "urgency": pri.get("urgency"),
            "historical_frequency": pri.get("historical_frequency"),
            "verification": ver,
            "raw": message,
            "generated_at": now_iso(),
            "needs_human_review": human_review
        }

        # Persist into long-term memory
        record = {
            "id": message.get("id", f"msg-{uuid.uuid4().hex}"),
            "text": message.get("text"),
            "ts": message.get("ts", now_iso()),
            "lat": geo.get("lat") if geo else None,
            "lon": geo.get("lon") if geo else None,
            "type": det.get("type"),
            "prob": det.get("prob"),
            "urgency": pri.get("urgency"),
            "verification_score": ver.get("image_verification", {}).get("match_score") if ver.get("image_verification") else None
        }
        try:
            self.memory_tool.call("insert_report", record)
        except Exception:
            logger.exception("Failed to persist memory record")

        # Reporter action (store JSON and summary)
        report_path = self.reporter.act(packet)

        return {"status": "disaster_detected", "packet": packet, "trace": trace, "report_path": report_path}

# -----------------------------------------------------------------------------
# CELL 8 — Instantiate agents for the demo
# -----------------------------------------------------------------------------
detector = SignalDetectionAgent()
geo_agent = GeoExtractionAgent(geotool)
verifier = VerificationAgent(img_tool)
priority_agent = PriorityAssessmentAgent(weathertool, memory_tool)
reporter = ReporterAgent(out_dir="reports")
coordinator = CoordinatorAgent(detector, verifier, geo_agent, priority_agent, reporter, memory_tool)

# -----------------------------------------------------------------------------
# CELL 9 — Demo dataset (sample messages) and stream replay
# -----------------------------------------------------------------------------
SAMPLE_MESSAGES = [
    {
        "id": "m1",
        "text": "Massive fire in Bandra West, buildings ablaze, people trapped!",
        "ts": (datetime.utcnow() - timedelta(minutes=30)).isoformat()
    },
    {
        "id": "m2",
        "text": "Water levels rising near Riverside Avenue; cars floating, people need rescue",
        "ts": (datetime.utcnow() - timedelta(minutes=25)).isoformat()
    },
    {
        "id": "m3",
        "text": "Strong tremors felt in Sector 21, buildings shaking - earthquake?",
        "ts": (datetime.utcnow() - timedelta(minutes=15)).isoformat()
    },
    {
        "id": "m4",
        "text": "Minor accident on Highway 5, multiple vehicles involved but no injuries reported",
        "ts": (datetime.utcnow() - timedelta(minutes=5)).isoformat()
    }
]

def replay_stream(messages: List[dict], delay: float = 0.5):
    for msg in messages:
        yield msg
        time.sleep(delay)

# Run demo replay and print outputs
print("\n=== Starting demo replay ===\n")
demo_results = []
for m in replay_stream(SAMPLE_MESSAGES, delay=0.5):
    try:
        out = coordinator.handle_message(m)
        demo_results.append(out)
        print("INPUT:", m["text"])
        pprint(out["packet"] if out.get("packet") else out)
        print("-" * 70)
    except Exception as e:
        logger.exception("Error processing message")
print("\n=== Demo complete ===\n")

# -----------------------------------------------------------------------------
# CELL 10 — Observability: Traces & Metrics
# -----------------------------------------------------------------------------
def compute_basic_metrics(engine):
    try:
        df = memory_tool.call("all_reports")
        if df is None or df.empty:
            print("No persisted reports yet.")
            return {}
        total = len(df)
        avg_urgency = float(df["urgency"].astype(float).mean())
        geocoded = df.dropna(subset=["lat", "lon"])
        geocode_rate = len(geocoded) / total if total > 0 else 0.0
        metrics = {
            "total_reports": total,
            "avg_urgency": avg_urgency,
            "geocode_rate": geocode_rate
        }
        print("Observability Metrics:")
        pprint(metrics)
        return metrics
    except Exception:
        logger.exception("Failed to compute metrics")
        return {}

metrics = compute_basic_metrics(engine)

# -----------------------------------------------------------------------------
# CELL 11 — Evaluation placeholders (use real datasets in final runs)
# -----------------------------------------------------------------------------
# This cell should be replaced by proper training + evaluation code using:
# - TREC-IS
# - CrisisLex
# - HumAID
# For quick demonstration we use a tiny holdout
from sklearn.metrics import precision_score, recall_score, f1_score

holdout = [
    ("House on fire near Church Lane", 1),
    ("Lovely weather today in my city", 0),
    ("Flood reported near riverside", 1),
]

X = [h[0] for h in holdout]
y = [h[1] for h in holdout]
probs = [detector.pipe.predict_proba([x])[0][1] for x in X]
preds = [1 if p >= 0.5 else 0 for p in probs]
print("Holdout evaluation (toy):")
print("Precision:", precision_score(y, preds))
print("Recall:", recall_score(y, preds))
print("F1:", f1_score(y, preds))

# -----------------------------------------------------------------------------
# CELL 12 — Export: Package the notebook outputs for submission
# -----------------------------------------------------------------------------
def package_submission(output_zip="LifeLine_submission_package.zip"):
    import zipfile
    roots = ["notebook.ipynb", "README.md", "submission_writeup.md", "video_script.txt", "requirements.txt"]
    # include modules and sample data + reports
    for dirpath in ["agents", "tools", "data", "reports"]:
        if os.path.exists(dirpath):
            for root, _, files in os.walk(dirpath):
                for fn in files:
                    fp = os.path.join(root, fn)
                    roots.append(fp)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in roots:
            if os.path.exists(p):
                zf.write(p)
    print("Created submission ZIP:", output_zip)

# (Optional) call package_submission() when ready
# package_submission()

# -----------------------------------------------------------------------------
# CELL 13 — Submission write-up & Video Script (copy into Kaggle submission)
# -----------------------------------------------------------------------------
SUBMISSION_WRITEUP = """
Title: LifeLine — ADK Multi-Agent Global Crisis Response

Abstract:
LifeLine is an ADK-first, multi-agent system that ingests social-media signals and auxiliary feeds,
detects and verifies crises, determines location and urgency, and generates rescue-ready dispatch
packets with evidence and metrics. The system demonstrates multi-agent orchestration, MCP-like
tool usage, session & long-term memory, observability (traces/logs/metrics), Agent2Agent (A2A)
messaging patterns, and a prototype deployment path.

Mapping to the 5-Day ADK Course:
Day 1 - Agents: Multi-agent decomposition (Detection, Verification, Geo, Triage, Reporter).
Day 2 - Tools & MCP: GeocodeTool, ImageVerificationTool, WeatherTool, MemoryTool.
Day 3 - Memory: Short-term traces in A2A messages + long-term SQLite memory.
Day 4 - Agent Quality: Logging, tracing, metrics, evaluation placeholders.
Day 5 - Prototype→Production: Packaging, Report generation, Streamlit demo provided in repo.

For code, datasets and demo instructions visit: [GitHub repo link placeholder]
"""

VIDEO_SCRIPT = """
LifeLine Demo (60-90s)
- Intro: "Hi, I'm [Your Name]. This is LifeLine..."
- Input: paste a sample tweet.
- Coordinator: show detection -> verification -> geo -> triage -> report.
- Show dispatch packet & map coordinates.
- Show observability metrics (geocode rate, avg latency).
- Close: link to repo & Kaggle submission.
"""

print("\n--- Submission writeup (ready to copy) ---")
print(SUBMISSION_WRITEUP)
print("\n--- Video script (ready to copy) ---")
print(VIDEO_SCRIPT)

# -----------------------------------------------------------------------------
# END OF NOTEBOOK
# -----------------------------------------------------------------------------
