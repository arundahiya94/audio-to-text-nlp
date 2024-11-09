import os
import logging
from pydub import AudioSegment
from pydub.utils import which

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Explicitly set paths to ffmpeg and ffprobe if not found by default
ffmpeg_path = which("ffmpeg")
ffprobe_path = which("ffprobe")

if not ffmpeg_path or not ffprobe_path:
    logger.error("ffmpeg or ffprobe not found. Ensure they are installed and accessible.")
    raise FileNotFoundError("ffmpeg or ffprobe are required but not found.")

AudioSegment.ffmpeg = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path

def convert_audio(input_path, output_path):
    try:
        # Check if the input file exists
        if not os.path.isfile(input_path):
            logger.error(f"Input file '{input_path}' does not exist.")
            raise FileNotFoundError(f"Input file '{input_path}' not found.")
        
        logger.info(f"Converting {input_path} to {output_path}")
        audio = AudioSegment.from_mp3(input_path)  # Load the MP3 file
        audio.export(output_path, format="wav")   # Convert and export as WAV
        logger.info(f"File converted successfully and saved to {output_path}")

    except Exception as e:
        logger.error(f"Error during audio conversion: {e}")
        raise

if __name__ == "__main__":
    input_audio_path = "D:/git/audio-to-text-nlp/07_christmasfantasy.mp3"
    output_audio_path = "converted_audio.wav"
    
    convert_audio(input_audio_path, output_audio_path)
