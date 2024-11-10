import re
import nltk
from nltk.corpus import stopwords
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data files (run this once)
nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocesses the transcription text by:
    - Lowercasing all text
    - Removing special characters and digits
    - Removing stopwords

    :param text: Raw transcription text.
    :return: Preprocessed text.
    """
    try:
        # Convert text to lowercase
        text = text.lower()
        logger.debug(f"Lowercased text: {text[:500]}...")  # Print first 500 characters for brevity

        # Remove special characters, numbers, and unwanted symbols
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        logger.debug(f"Text after removing special characters: {text[:500]}...")

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in text.split() if word not in stop_words]
        processed_text = " ".join(filtered_words)
        
        logger.info("Text preprocessing completed.")
        logger.info(f"Final number of words after processing: {len(filtered_words)}")
        return processed_text

    except Exception as e:
        logger.error(f"Error during text preprocessing: {e}")
        raise

def load_transcription(file_path):
    """
    Loads the transcription text from a file.
    
    :param file_path: Path to the transcription file.
    :return: The raw text from the file.
    """
    try:
        with open(file_path, 'r') as file:
            text = file.read()
        logger.info(f"Loaded transcription from {file_path}")
        return text
    except Exception as e:
        logger.error(f"Error loading transcription from {file_path}: {e}")
        raise
