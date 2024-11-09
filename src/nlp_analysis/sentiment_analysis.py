import logging
from transformers import pipeline
from nlp_analysis.preprocessing import load_transcription, preprocess_text

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def perform_sentiment_analysis(file_path):
    """
    Load, preprocess, and analyze the sentiment of a transcription text.
    
    :param file_path: Path to the transcription file.
    :return: Sentiment analysis result if successful, None otherwise.
    """
    try:
        # Load the transcription text
        transcription = load_transcription(file_path)
        if not transcription:
            logger.error("Transcription file is empty or could not be loaded.")
            return None

        # Preprocess the transcription text
        processed_text = preprocess_text(transcription)
        
        # Perform sentiment analysis using Hugging Face's pre-trained model
        logger.info("Starting sentiment analysis...")
        sentiment_analyzer = pipeline("sentiment-analysis")
        sentiment = sentiment_analyzer(processed_text)

        logger.info(f"Sentiment analysis completed: {sentiment}")
        return sentiment

    except Exception as e:
        logger.error(f"An error occurred during sentiment analysis: {e}")
        return None

if __name__ == "__main__":
    transcription_file_path = "data/transcription_output.txt"
    result = perform_sentiment_analysis(transcription_file_path)

    if result:
        print(f"Sentiment: {result}")
    else:
        print("Sentiment analysis could not be completed.")
