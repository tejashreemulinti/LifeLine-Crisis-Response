import requests
import os

class GeoAgent:
    def geocode(self, text):
        key = os.getenv("GOOGLE_API_KEY")
        url = f"https://maps.googleapis.com/maps/api/geocode/json?address={text}&key={key}"

        r = requests.get(url)
        return r.json()
