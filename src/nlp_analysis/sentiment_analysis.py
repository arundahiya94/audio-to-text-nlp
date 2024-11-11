from transformers import pipeline, AutoTokenizer
import logging
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the sentiment analyzer and tokenizer
sentiment_analyzer = pipeline("sentiment-analysis")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def perform_sentiment_analysis(text):
    """
    Performs sentiment analysis on text.
    
    :param text: Preprocessed text.
    :return: Sentiment result with label and score.
    """
    return perform_sentiment_analysis_sliding_window(text=text)

def perform_sentiment_analysis_sliding_window(text, window_size=512, overlap=256):
    """
    Performs sentiment analysis on text using a sliding window approach.
    
    :param text: Preprocessed text.
    :param window_size: Maximum number of tokens for the model.
    :param overlap: Number of overlapping tokens between consecutive chunks.
    :return: Aggregated sentiment for the entire document.
    """
    logger.info("Analyzing sentiment for the provided text using sliding window approach.")
    
    # Tokenize the entire text
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"].squeeze().tolist()
    num_tokens = len(tokens)
    chunk_scores = []
    
    start = 0
    while start < num_tokens:
        end = min(start + window_size, num_tokens)
        chunk_tokens = tokens[start:end]
        
        # Decode chunk tokens back to text for sentiment analysis
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        # Perform sentiment analysis on the current chunk
        sentiment = sentiment_analyzer(chunk_text, truncation=True)[0]
        chunk_scores.append((sentiment['label'], sentiment['score']))
        
        # Log each chunk's sentiment result
        logger.info(f"Chunk ({start}:{end}) - Label: {sentiment['label']}, Score: {sentiment['score']}")
        
        # Move the start position forward, considering overlap
        start += window_size - overlap
    
    # Plot the sentiment scores for each chunk
    plot_sentiment_scores(chunk_scores)
    
    # Aggregate the overall sentiment by majority vote
    positive_count = sum(1 for label, _ in chunk_scores if label == "POSITIVE")
    negative_count = sum(1 for label, _ in chunk_scores if label == "NEGATIVE")
    
    overall_sentiment = "POSITIVE" if positive_count > negative_count else "NEGATIVE" if negative_count > positive_count else "NEUTRAL"
    logger.info(f"Overall Document Sentiment: {overall_sentiment}")
    
    return overall_sentiment

def plot_sentiment_scores(chunk_scores):
    """
    Plots the sentiment scores for each chunk.

    :param chunk_scores: List of tuples (label, score).
    """
    # Generate chunk labels based on the number of scores
    chunk_labels = [f"Chunk {i+1}" for i in range(len(chunk_scores))]
    positive_scores = [score if label == 'POSITIVE' else 0 for label, score in chunk_scores]
    negative_scores = [score if label == 'NEGATIVE' else 0 for label, score in chunk_scores]

    plt.figure(figsize=(10, 6))
    plt.plot(chunk_labels, positive_scores, label="Positive", color="green", marker="o")
    plt.plot(chunk_labels, negative_scores, label="Negative", color="red", marker="o")
    plt.xlabel("Chunks")
    plt.ylabel("Sentiment Score")
    plt.title("Sentiment Scores for Text Chunks")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

