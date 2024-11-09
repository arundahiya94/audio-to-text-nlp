from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def perform_topic_modeling(text, num_topics=3):
    # Tokenize and vectorize the text for LDA
    vectorizer = CountVectorizer(stop_words='english')
    text_vectorized = vectorizer.fit_transform([text])
    
    # Fit LDA model
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(text_vectorized)
    
    # Extract topics and their top words
    words = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[:-6:-1]]
        topics.append(f"Topic {topic_idx + 1}: {' '.join(top_words)}")
    
    return topics
