from transformers import pipeline

def perform_sentiment_analysis(text):
    sentiment_analyzer = pipeline("sentiment-analysis")
    sentiment = sentiment_analyzer(text)
    return sentiment
