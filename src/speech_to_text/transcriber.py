import assemblyai as aai

# Replace with your API key
aai.settings.api_key = "ee0d473b0e3b4a34ac2428ab5c3ddaa4"

# URL of the file to transcribe
FILE_URL = "D:/git/audio-to-text-nlp/converted_audio.wav"

# You can also transcribe a local file by passing in a file path
# FILE_URL = './path/to/file.mp3'

transcriber = aai.Transcriber()
transcript = transcriber.transcribe(FILE_URL)

if transcript.status == aai.TranscriptStatus.error:
    print(transcript.error)
else:
    # Save the transcript text to a file
    with open("transcription_output.txt", "w") as text_file:
        text_file.write(transcript.text)
    print("Transcription saved to 'transcription_output.txt'")
