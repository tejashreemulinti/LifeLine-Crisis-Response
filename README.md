# LifeLine — Multi-Agent Crisis Response System (ADK-Based)

LifeLine is an advanced multi-agent emergency response system built using the
Google **Agent Development Kit (ADK)**.

This project is designed for the **Kaggle AI Agents Intensive Capstone 2025** and
demonstrates:

✔ Multi-agent architecture  
✔ ADK tools  
✔ MCP-style interoperability  
✔ Long-term + short-term memory  
✔ A2A protocol  
✔ Observability, logging & evaluation  
✔ Real-world problem-solving (emergency detection & reporting)

---

## 🔥 Agents Included

### 1. DetectionAgent

Classifies crisis messages (accident, fire, flood, injury, etc.)

### 2. VerificationAgent

Double-checks via second-pass validation.  
Prevents false positives.

### 3. GeoAgent

Uses Google Maps Geocoding API to convert text → coordinates.

### 4. CoordinatorAgent

Combines all agent outputs and builds final structured incident report.

### 5. ReporterAgent

Saves reports to `/reports/` & prints summary.

---

## 🛠 Tools

| Tool             | Purpose                                   |
| ---------------- | ----------------------------------------- |
| GeocodeTool      | Google Geocoding API wrapper              |
| MemoryTool       | Simple long-term memory bank (JSON store) |
| WeatherTool      | Optional future extension                 |
| VerificationTool | LLM-based text verification               |

---

## 🚀 Running Locally

Set your API key:

Windows (PowerShell):

```
$env:GOOGLE_API_KEY="YOUR_KEY"
```

Then run:

```
python notebook.py
```

---

## 📂 Folder Structure

(As shown above)

---

## 📘 License

MIT
