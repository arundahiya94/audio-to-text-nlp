from transformers import pipeline, AutoTokenizer
import logging
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the sentiment analyzer and tokenizer
sentiment_analyzer = pipeline("sentiment-analysis")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

def split_text_into_chunks(text, max_tokens=512):
    """
    Splits the text into chunks based on token count, ensuring that no chunk exceeds max_tokens.
    """
    tokens = tokenizer.tokenize(text)
    chunk_size = max_tokens - 2  # Reserve space for special tokens
    chunks = []
    current_chunk = []

    for token in tokens:
        current_chunk.append(token)
        if len(current_chunk) >= chunk_size:
            chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
            current_chunk = []

    if current_chunk:  # Add remaining tokens as a final chunk
        chunks.append(tokenizer.convert_tokens_to_string(current_chunk))

    logger.info(f"Text split into {len(chunks)} chunks.")
    return chunks

def perform_sentiment_analysis(text):
    """
    Performs sentiment analysis on text, handling long text by chunking if necessary.
    
    :param text: Preprocessed text.
    :return: List of sentiment results for each chunk.
    """
    chunks = split_text_into_chunks(text)
    sentiment_results = []
    chunk_scores = []  # Store scores for visualization

    for i, chunk in enumerate(chunks, 1):
        logger.info(f"Analyzing sentiment for chunk {i}/{len(chunks)}")
        sentiment = sentiment_analyzer(chunk)[0]
        sentiment_results.append({
            "chunk": i,
            "label": sentiment['label'],
            "score": sentiment['score']
        })
        chunk_scores.append((sentiment['label'], sentiment['score']))  # Collect data for visualization

    # Log results
    for result in sentiment_results:
        logger.info(f"Chunk {result['chunk']}: Label - {result['label']}, Score - {result['score']}")

    # Plot sentiment scores
    plot_sentiment_scores(chunk_scores)

    return sentiment_results

def plot_sentiment_scores(chunk_scores):
    """
    Plots the sentiment scores for each chunk.

    :param chunk_scores: List of tuples (label, score) for each chunk.
    """
    chunk_labels = [f"Chunk {i+1}" for i in range(len(chunk_scores))]
    positive_scores = [score if label == 'POSITIVE' else 0 for label, score in chunk_scores]
    negative_scores = [score if label == 'NEGATIVE' else 0 for label, score in chunk_scores]

    plt.figure(figsize=(10, 6))
    plt.plot(chunk_labels, positive_scores, label="Positive", color="green", marker="o")
    plt.plot(chunk_labels, negative_scores, label="Negative", color="red", marker="o")
    plt.xlabel("Chunks")
    plt.ylabel("Sentiment Score")
    plt.title("Sentiment Scores per Chunk")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
