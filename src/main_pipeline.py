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
    # Tokenize the text
    tokenized_text = tokenizer.encode(text, truncation=False)
    chunks = []
    
    # Split tokenized text into chunks of size max_tokens or less
    for i in range(0, len(tokenized_text), max_tokens):
        chunk = tokenized_text[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        chunks.append(chunk_text)

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

        logger.info(f"\n===== Starting Text Preprocessing =====\n ")
        processed_text = preprocess_text(transcription) 
        logger.info(f"\n===== Ending Text Preprocessing =====\n ")

        # Step 1: Summarization
        try:
            logger.info(f"\n===== Summary Start =====\n")
            chunks = split_text_for_summary(processed_text, max_tokens=1024)
            
            # Generate summary for each chunk
            summary_parts = [generate_summary(chunk) for chunk in chunks]
            
            # Join summaries from each chunk into one final summary
            summary = " ".join(summary_parts)
            logger.info(f"{summary}")
            logger.info(f"\n===== Summary Results =====\n")
        except Exception as e:
            logger.error(f"Error in summarization: {e}")

        # Step 2: Topic Modeling
        try:
            logger.info(f"\n===== Topic Modeling Start =====\n")
            topics = perform_topic_modeling(transcription)
            logger.info(f"{topics}")
            logger.info(f"\n===== Topic Modeling Results =====\n")
        except Exception as e:
            logger.error(f"Error in topic modeling: {e}")

        # Step 3: Sentiment Analysis
        try:
            logger.info(f"\n===== Sentiment Analysis Start =====\n")
            sentiment_results = perform_sentiment_analysis(processed_text)
            logger.info(f"Over all Sentiments : {sentiment_results}")
            logger.info(f"\n===== Sentiment Analysis Results =====\n ")
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")

    except Exception as e:
        logger.error(f"Error in pipeline setup or preprocessing: {e}")

if __name__ == "__main__":
    # Define the path relative to the current script's location
    transcription_file_path = os.path.join("..", "data", "transcription_output.txt")
    
    logger.info(f"\n===== Pipeline Started =====\n ")
    run_pipeline(transcription_file_path)
    logger.info(f"\n===== Pipeline Ended =====\n ")
