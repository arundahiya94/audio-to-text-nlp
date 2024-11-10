# summarization.py
from transformers import pipeline

def generate_summary(text):
    """
    Summarizes a given text using a pre-trained summarization model.
    
    :param text: Input text to summarize.
    :return: Summary of the text.
    """
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    # summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

