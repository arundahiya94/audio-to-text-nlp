from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np

def summarize_text(text, num_sentences=3):
    """
    Summarizes the text by selecting the top N sentences based on their relevance.

    :param text: Text to summarize.
    :param num_sentences: Number of sentences to include in the summary.
    :return: Summary of the text.
    """
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
    
    # Visualize sentence scores with a bar chart
    visualize_sentence_scores(sentences, sentence_scores, ranked_sentences_idx)
    
    return ' '.join(summary)

def visualize_sentence_scores(sentences, sentence_scores, ranked_sentences_idx):
    """
    Visualizes sentence scores using a bar chart, highlighting selected sentences for the summary.

    :param sentences: List of sentences from the text.
    :param sentence_scores: Array of similarity scores for each sentence.
    :param ranked_sentences_idx: Indices of sentences selected for the summary.
    """
    # Set up labels for bar chart
    sentence_labels = [f"Sentence {i+1}" for i in range(len(sentences))]
    colors = ["orange" if i in ranked_sentences_idx else "skyblue" for i in range(len(sentences))]
    
    # Plot bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(sentence_labels, sentence_scores, color=colors)
    plt.xlabel("Sentences")
    plt.ylabel("Sentence Score")
    plt.title("Sentence Scores for Summarization")
    
    # Highlight sentences selected for summary
    for bar, idx in zip(bars, range(len(sentence_scores))):
        if idx in ranked_sentences_idx:
            bar.set_edgecolor("red")
            bar.set_linewidth(2)
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
