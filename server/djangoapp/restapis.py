import requests
import os
from pathlib import Path
from dotenv import load_dotenv
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

load_dotenv()

backend_url = os.getenv(
    "backend_url", default="http://localhost:3030"
)

# Make the bundled VADER lexicon available to NLTK.
NLTK_DATA_DIR = Path(__file__).resolve().parent / "microservices"
if str(NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.append(str(NLTK_DATA_DIR))

sia = SentimentIntensityAnalyzer()


def get_request(endpoint, **kwargs):
    params = ""
    if kwargs:
        for key, value in kwargs.items():
            params = params + key + "=" + str(value) + "&"

    request_url = backend_url + endpoint + "?" + params

    print("GET from {} ".format(request_url))
    try:
        response = requests.get(request_url)
        return response.json()
    except requests.RequestException:
        print("Network exception occurred")


def analyze_review_sentiments(text):
    scores = sia.polarity_scores(text)
    pos = float(scores["pos"])
    neg = float(scores["neg"])
    neu = float(scores["neu"])

    sentiment = "positive"
    if neg > pos and neg > neu:
        sentiment = "negative"
    elif neu > neg and neu > pos:
        sentiment = "neutral"

    return {"sentiment": sentiment}


def post_review(data_dict):
    request_url = backend_url + "/insert_review"
    try:
        response = requests.post(request_url, json=data_dict)
        print(response.json())
        return response.json()
    except requests.RequestException:
        print("Network exception occurred")
