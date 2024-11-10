import logging
from nlp_analysis.preprocessing import load_transcription, preprocess_text
from nlp_analysis.sentiment_analysis import perform_sentiment_analysis
from nlp_analysis.topic_modelling import perform_topic_modeling
from nlp_analysis.summarization import generate_summary
import os
from transformers import AutoTokenizer


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize tokenizer with the same model you’re using for summarization
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

def split_text_for_summary(text, max_tokens=1024):
    """
    Splits the text into chunks that fit within the model's maximum token length.

    :param text: Preprocessed text.
    :param max_tokens: Maximum number of tokens allowed per chunk.
    :return: List of text chunks.
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    chunk_texts = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
    return chunk_texts

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

        logger.info(f"\n===== Starting Text Preprocessing =====\n ")
        processed_text = preprocess_text(transcription) 
        logger.info(f"\n===== Ending Text Preprocessing =====\n ")

        # Step 3: Summarization
        # chunks = split_text_for_summary(processed_text)
        # summary_parts = [generate_summary(chunk) for chunk in chunks]
        summary_parts = generate_summary(processed_text)
        summary = " ".join(summary_parts)
        
        logger.info(f"\n===== Summary Results =====\n {summary}")

        # Step 2: Topic Modeling
        topics = perform_topic_modeling(processed_text)
        logger.info(f"\n===== Topic Modeling Results =====\n {topics}")

        # # Step 1: Sentiment Analysis
        # sentiment_results = perform_sentiment_analysis(processed_text)
        # logger.info(f"\n===== Sentiment Analysis Results =====\n {sentiment_results}")

    except Exception as e:
        logger.error(f"Error in pipeline: {e}")

if __name__ == "__main__":
    # Define the path relative to the current script's location
    transcription_file_path = os.path.join("..", "data", "transcription_output.txt")
    
    logger.info(f"\n===== Pipeline Started =====\n ")
    run_pipeline(transcription_file_path)
    logger.info(f"\n===== Pipeline Ended =====\n ")
