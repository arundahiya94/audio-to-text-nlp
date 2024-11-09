import logging
import os
from dotenv import load_dotenv
import assemblyai as aai

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from a .env file
load_dotenv()  # This loads the variables from .env into the environment

# Retrieve API Key from environment variables
API_KEY = os.getenv("ASSEMBLYAI_API_KEY")  # Fetch the API key securely from the environment
if not API_KEY:
    logger.error("API Key for AssemblyAI is missing. Please set the environment variable.")
    raise ValueError("AssemblyAI API Key is required.")

aai.settings.api_key = API_KEY

def transcribe_audio(file_path):
    try:
        # Check if the file exists
        if not os.path.isfile(file_path):
            logger.error(f"Audio file '{file_path}' does not exist.")
            raise FileNotFoundError(f"Audio file '{file_path}' not found.")

        # Transcribe the audio file
        logger.info(f"Starting transcription for {file_path}")
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(file_path)

        # Check if there was an error with the transcription
        if transcript.status == aai.TranscriptStatus.error:
            logger.error(f"Error transcribing the audio file: {transcript.error}")
            return

        # Updated path to save transcription output
        output_file = os.path.join("..", "data", "transcription_output.txt")
        
        # Ensure the 'data' folder exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)  
        
        with open(output_file, "w") as text_file:
            text_file.write(transcript.text)
        
        logger.info(f"Transcription saved to '{output_file}'")

    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise

if __name__ == "__main__":

    audio_file_path = os.path.join("..", "data", "converted_audio.wav")
    
    transcribe_audio(audio_file_path)
