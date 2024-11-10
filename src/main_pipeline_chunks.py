import math
import logging
import nltk
from nlp_analysis.preprocessing import load_transcription, preprocess_text
from nlp_analysis.sentiment_analysis import perform_sentiment_analysis
from nlp_analysis.topic_modelling import perform_topic_modeling
from nlp_analysis.summarization import generate_summary
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer  # Added to load tokenizer
import os

nltk.download('punkt')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the tokenizer for the model you are using (replace with your specific model)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

def split_text_into_chunks(processed_text, max_tokens=512):
    """
    Splits the text into chunks based on token count, ensuring that no chunk exceeds max_tokens.
    This method accounts for special tokens that might be added during tokenization.

    :param processed_text: Preprocessed text.
    :param max_tokens: Maximum number of tokens allowed per chunk (including special tokens).
    :return: List of text chunks.
    """
    # Tokenize using the model's tokenizer
    tokens = tokenizer.tokenize(processed_text)
    logger.info(f"Total tokens before splitting: {len(tokens)}")
    
    # Account for special tokens
    chunk_size = max_tokens - 2  # Reserve 2 tokens for special tokens like [CLS] and [SEP]
    chunks = []
    current_chunk = []

    for token in tokens:
        current_chunk.append(token)
        if len(current_chunk) >= chunk_size:
            chunks.append(" ".join(current_chunk))  # Add current chunk to list
            current_chunk = []  # Reset for the next chunk

    if current_chunk:  # If there are remaining tokens, add them as a final chunk
        chunks.append(" ".join(current_chunk))

    logger.info(f"Text split into {len(chunks)} chunks.")
    return chunks


def run_pipeline(file_path):
    """
    Main pipeline to process the transcription and perform sentiment analysis, topic modeling, and summarization.
    
    :param file_path: Path to the transcription file.
    """
    try:
        # Load and preprocess transcription
        transcription = load_transcription(file_path)
        if not transcription:
            logger.error("Transcription file is empty or could not be loaded.")
            return

        processed_text = preprocess_text(transcription)

        # Step 1: Chunk the processed text into smaller chunks
        chunks = split_text_into_chunks(processed_text)
        logger.info(f"Text split into {len(chunks)} chunks.")

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            logger.info(f"Number of tokens in chunk {i+1}: {len(word_tokenize(chunk))}")

            # Step 2: Sentiment Analysis for each chunk
            sentiment = perform_sentiment_analysis(chunk)
            logger.info(f"Sentiment Analysis Result for chunk {i+1}: {sentiment}")

            # Step 3: Topic Modeling for each chunk
            topics = perform_topic_modeling(chunk)
            logger.info(f"Identified Topics for chunk {i+1}: {topics}")

            # Step 4: Summarization for each chunk
            summary = generate_summary(chunk)
            logger.info(f"Text Summary for chunk {i+1}: {summary}")

    except Exception as e:
        logger.error(f"Error in pipeline: {e}")



if __name__ == "__main__":
    # Define the path relative to the current script's location
    transcription_file_path = os.path.join("..", "data", "transcription_output.txt")
    
    run_pipeline(transcription_file_path)
