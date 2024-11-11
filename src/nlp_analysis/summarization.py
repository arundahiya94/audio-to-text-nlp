from transformers import pipeline
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def generate_summary(text):
    """
    Summarizes a given text using a pre-trained summarization model.
    
    :param text: Input text to summarize.
    :return: Summary of the text.
    """
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    # Ensure the text is within the model's token limit
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']


def summarize_text(text, num_sentences=3):
    # Step 1: Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Step 2: Compute the TF-IDF matrix for the sentences
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    
    # Step 3: Compute the cosine similarity between all sentence pairs
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Step 4: Rank the sentences based on the sum of cosine similarities
    sentence_scores = cosine_sim.sum(axis=1)
    
    # Step 5: Select the top `num_sentences` sentences with the highest scores
    ranked_sentences_idx = sentence_scores.argsort()[-num_sentences:][::-1]
    
    # Create the summary by selecting the top-ranked sentences
    summary = [sentences[idx] for idx in ranked_sentences_idx]
    
    return ' '.join(summary)