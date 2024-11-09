import logging
from nlp_analysis.preprocessing import load_transcription, preprocess_text
from nlp_analysis.sentiment_analysis import perform_sentiment_analysis
from nlp_analysis.topic_modelling import perform_topic_modeling
from nlp_analysis.summarization import generate_summary
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_text_into_chunks(text, max_chunk_length=512):
    """
    Splits the text into smaller chunks to avoid exceeding model input limits.
    
    :param text: The raw text to be split.
    :param max_chunk_length: Maximum length of each chunk (default is 512 tokens, but it can vary based on the model).
    :return: A list of text chunks.
    """
    # Split the text into sentences or a smaller unit to make chunking more coherent
    sentences = text.split('. ')  # Simple sentence splitting, can be replaced with more sophisticated tokenization

    chunks = []
    current_chunk = []

    for sentence in sentences:
        if len(' '.join(current_chunk + [sentence])) <= max_chunk_length:
            current_chunk.append(sentence)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
    
    # Add the last chunk if there's any content left
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
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

        # Preprocess the transcription text
        processed_text = preprocess_text(transcription)

        # Split the processed text into chunks
        chunks = split_text_into_chunks(processed_text)

        # Process each chunk individually
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")

            # Step 1: Sentiment Analysis
            sentiment = perform_sentiment_analysis(chunk)
            logger.info(f"Sentiment Analysis Result for chunk {i+1}: {sentiment}")

            # Step 2: Topic Modeling
            topics = perform_topic_modeling(chunk)
            logger.info(f"Identified Topics for chunk {i+1}: {topics}")

            # Step 3: Summarization
            summary = generate_summary(chunk)
            logger.info(f"Text Summary for chunk {i+1}: {summary}")

    except Exception as e:
        logger.error(f"Error in pipeline: {e}")

if __name__ == "__main__":
    # Define the path relative to the current script's location
    transcription_file_path = os.path.join("..", "data", "transcription_output.txt")
    
    run_pipeline(transcription_file_path)
