import google.generativeai as genai

class DetectionAgent:
    def __init__(self, model="gemini-1.5-flash"):
        self.model = genai.GenerativeModel(model)

    def classify(self, text):
        prompt = f"""
        Classify the following message into one emergency category:
        Options: [fire, flood, accident, medical, crime, earthquake, other]

        Message: {text}

        Respond in JSON:
        {{
            "category": "...",
            "confidence": "0-1",
            "rationale": "..."
        }}
        """

        resp = self.model.generate_content(prompt)
        return resp.text
