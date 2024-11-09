from nlp_analysis.preprocessing import load_transcription, preprocess_text
from transformers import pipeline

# Load the transcription text
transcription = load_transcription('data/transcription_output.txt')

# Preprocess the transcription text
processed_text = preprocess_text(transcription)

# Perform sentiment analysis using Hugging Face's pre-trained model
sentiment_analyzer = pipeline("sentiment-analysis")
sentiment = sentiment_analyzer(processed_text)

print(f"Sentiment: {sentiment}")
