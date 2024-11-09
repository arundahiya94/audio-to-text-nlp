from pydub import AudioSegment
from pydub.utils import which

# Explicitly set paths to ffmpeg and ffprobe if not found by default
AudioSegment.ffmpeg = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

# Now attempt to load and convert the file
audio = AudioSegment.from_mp3("D:/git/audio-to-text-nlp/07_christmasfantasy.mp3")
audio.export("converted_audio.wav", format="wav")

print("File converted successfully")
