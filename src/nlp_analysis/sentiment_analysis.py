from transformers import pipeline, AutoTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the sentiment analyzer and tokenizer
sentiment_analyzer = pipeline("sentiment-analysis")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

def split_text_into_chunks(text, max_tokens=512):
    """
    Splits the text into chunks based on token count, ensuring that no chunk exceeds max_tokens.

    :param text: Preprocessed text.
    :param max_tokens: Maximum number of tokens allowed per chunk.
    :return: List of text chunks.
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

    for i, chunk in enumerate(chunks, 1):
        logger.info(f"Analyzing sentiment for chunk {i}/{len(chunks)}")
        sentiment = sentiment_analyzer(chunk)
        sentiment_results.append(sentiment)

    return sentiment_results
