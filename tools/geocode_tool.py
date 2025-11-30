import requests, os

def geocode(address):
    key = os.getenv("GOOGLE_API_KEY")
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={key}"
    return requests.get(url).json()
