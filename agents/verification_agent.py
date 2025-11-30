import google.generativeai as genai

class VerificationAgent:
    def __init__(self, model="gemini-1.5-flash"):
        self.model = genai.GenerativeModel(model)

    def verify(self, text, category):
        prompt = f"""
        Double-check this classification:

        Message: {text}
        Proposed Category: {category}

        Respond in JSON:
        {{
            "verified": true/false,
            "reason": "..."
        }}
        """

        resp = self.model.generate_content(prompt)
        return resp.text
