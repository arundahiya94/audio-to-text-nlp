from transformers import pipeline, AutoTokenizer
import logging
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the transformer-based sentiment analyzer, tokenizer, and VADER sentiment analyzer
transformer_sentiment_analyzer = pipeline("sentiment-analysis")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
vader_analyzer = SentimentIntensityAnalyzer()

def perform_sentiment_analysis(text):
    """
    Performs sentiment analysis on text using both transformer-based model and VADER.
    """
    return perform_sentiment_analysis_sliding_window(text=text)

def perform_sentiment_analysis_sliding_window(text, window_size=512, overlap=256):
    """
    Performs sentiment analysis on text using a sliding window approach and adds VADER sentiment.
    
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
        
        # Perform sentiment analysis on the current chunk using transformer model
        transformer_sentiment = transformer_sentiment_analyzer(chunk_text, truncation=True)[0]
        
        # Perform sentiment analysis on the current chunk using VADER
        vader_sentiment = vader_analyzer.polarity_scores(chunk_text)
        
        # Collect and log the combined sentiment result
        chunk_scores.append({
            "label": transformer_sentiment['label'],
            "score": transformer_sentiment['score'],
            "vader_pos": vader_sentiment['pos'],
            "vader_neu": vader_sentiment['neu'],
            "vader_neg": vader_sentiment['neg'],
            "vader_compound": vader_sentiment['compound']
        })
        
        logger.info(
            f"Chunk ({start}:{end}) - Transformer Label: {transformer_sentiment['label']}, "
            f"Score: {transformer_sentiment['score']}, VADER Pos: {vader_sentiment['pos']}, "
            f"Neu: {vader_sentiment['neu']}, Neg: {vader_sentiment['neg']}, Compound: {vader_sentiment['compound']}"
        )
        
        # Move the start position forward, considering overlap
        start += window_size - overlap

    # Plot sentiment scores for each chunk
    plot_sentiment_scores(chunk_scores)
    
    # Aggregate overall sentiment based on transformer labels
    positive_count = sum(1 for score in chunk_scores if score["label"] == "POSITIVE")
    negative_count = sum(1 for score in chunk_scores if score["label"] == "NEGATIVE")
    overall_sentiment = (
        "POSITIVE" if positive_count > negative_count 
        else "NEGATIVE" if negative_count > positive_count 
        else "NEUTRAL"
    )
    
    logger.info(f"Overall Document Sentiment: {overall_sentiment}")
    
    return overall_sentiment

def plot_sentiment_scores(chunk_scores):
    """
    Plots the sentiment scores for each chunk, including VADER scores.
    """
    # Generate chunk labels
    chunk_labels = [f"Chunk {i+1}" for i in range(len(chunk_scores))]
    
    # Extract individual scores
    transformer_positive_scores = [
        score["score"] if score["label"] == 'POSITIVE' else 0 for score in chunk_scores
    ]
    transformer_negative_scores = [
        score["score"] if score["label"] == 'NEGATIVE' else 0 for score in chunk_scores
    ]
    vader_positive = [score['vader_pos'] for score in chunk_scores]
    vader_neutral = [score['vader_neu'] for score in chunk_scores]
    vader_negative = [score['vader_neg'] for score in chunk_scores]
    vader_compound = [score['vader_compound'] for score in chunk_scores]

    # Plot transformer model scores
    plt.figure(figsize=(12, 8))
    plt.plot(chunk_labels, transformer_positive_scores, label="Transformer Positive", color="green", marker="o")
    plt.plot(chunk_labels, transformer_negative_scores, label="Transformer Negative", color="red", marker="o")
    
    # Plot VADER sentiment scores
    plt.plot(chunk_labels, vader_positive, label="VADER Positive", color="lightgreen", linestyle="--")
    plt.plot(chunk_labels, vader_neutral, label="VADER Neutral", color="blue", linestyle="--")
    plt.plot(chunk_labels, vader_negative, label="VADER Negative", color="orange", linestyle="--")
    plt.plot(chunk_labels, vader_compound, label="VADER Compound", color="purple", linestyle="--")
    
    # Customize plot
    plt.xlabel("Chunks")
    plt.ylabel("Sentiment Score")
    plt.title("Sentiment Scores for Text Chunks")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
