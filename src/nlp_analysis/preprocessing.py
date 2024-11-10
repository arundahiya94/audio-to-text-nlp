import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data files (run this once)
nltk.download('stopwords')
nltk.download('punkt')

# Initialize spell checker
spell = SpellChecker()

def preprocess_text(text):
    """
    Preprocesses the transcription text by:
    - Lowercasing all text
    - Removing special characters and digits
    - Tokenizing text
    - Removing stopwords
    - Correcting spelling

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

        # Tokenize text into words
        tokens = word_tokenize(text)
        logger.debug(f"Tokenized text: {tokens[:5]}...")  # Print first 5 tokens for brevity
        logger.info(f"Number of tokens after tokenization: {len(tokens)}")

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        logger.debug(f"Tokens after stopword removal: {filtered_tokens[:5]}...")  # Print first 5 tokens for brevity
        logger.info(f"Number of tokens after stopword removal: {len(filtered_tokens)}")

        # Correct spelling (commented out for now)
        # corrected_tokens = [spell.correction(word) for word in filtered_tokens if word is not None]
        # logger.debug(f"Tokens after spelling correction: {corrected_tokens[:5]}...")  # Print first 5 tokens
        # logger.info(f"Number of tokens after spelling correction: {len(corrected_tokens)}")

        # Join tokens back into a single string
        processed_text = " ".join(filtered_tokens)
        
        logger.info("Text preprocessing completed.")
        logger.info(f"Final number of tokens after processing: {len(filtered_tokens)}")
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
