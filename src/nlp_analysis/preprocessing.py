import re
import nltk
from nltk.corpus import stopwords
import logging
from nltk.tokenize import word_tokenize


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure stopwords are downloaded (only need to run once)
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocesses the transcription text by:
    - Tokenizing the text
    - Lowercasing all text
    - Removing special characters and digits
    - Removing stopwords
    - Rejoining tokens for further processing (if needed)

    :param text: Raw transcription text.
    :return: Preprocessed text as a single string.
    """
    try:
        # Tokenize the text
        tokens = word_tokenize(text)
        logger.debug(f"Initial tokens: {tokens[:20]}...")  # Display first 20 tokens for brevity

        # Convert tokens to lowercase
        tokens = [token.lower() for token in tokens]
        logger.debug(f"Lowercased tokens: {tokens[:20]}...")

        # Remove special characters and digits
        tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens]
        tokens = [token for token in tokens if token]  # Remove any empty tokens
        logger.debug(f"Tokens after removing special characters: {tokens[:20]}...")

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token not in stop_words]
        logger.debug(f"Tokens after stopword removal: {filtered_tokens[:20]}...")

        # Rejoin tokens into a single string for sentiment analysis
        processed_text = " ".join(filtered_tokens)
        
        logger.info("Text preprocessing completed.")
        logger.info(f"Final number of words after processing: {len(filtered_tokens)}")
        logger.info(f"Preprocessed text: {processed_text[:500]}...")  # Print first 500 characters for brevity

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
