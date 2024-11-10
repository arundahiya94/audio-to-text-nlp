import logging
from nlp_analysis.preprocessing import load_transcription, preprocess_text
from nlp_analysis.sentiment_analysis import perform_sentiment_analysis
from nlp_analysis.topic_modelling import perform_topic_modeling
from nlp_analysis.summarization import generate_summary
from transformers import AutoTokenizer  # Added to load tokenizer
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

        # Initialize tokenizer (using distilbert as an example)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        # Tokenize processed text and print the number of tokens
        tokenized_input = tokenizer(processed_text, truncation=True, padding=True, return_tensors="pt")
        num_tokens = tokenized_input["input_ids"].shape[1]  # Get the number of tokens
        logger.info(f"Number of tokens: {num_tokens}")

        # Step 1: Sentiment Analysis
        sentiment = perform_sentiment_analysis(processed_text)
        logger.info(f"Sentiment Analysis Result: {sentiment}")

        # Step 2: Topic Modeling
        topics = perform_topic_modeling(processed_text)
        logger.info(f"Identified Topics: {topics}")

        # Step 3: Summarization
        summary = generate_summary(processed_text)
        logger.info(f"Text Summary: {summary}")

    except Exception as e:
        logger.error(f"Error in pipeline: {e}")

if __name__ == "__main__":
    # Define the path relative to the current script's location
    transcription_file_path = os.path.join("..", "data", "transcription_output.txt")
    
    run_pipeline(transcription_file_path)
