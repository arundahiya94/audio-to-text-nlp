import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Load spaCy model for lemmatization and stopword removal
nlp = spacy.load("en_core_web_sm")

def perform_topic_modeling(text, num_topics=3):
    """
    Performs topic modeling on the input text using LDA (Latent Dirichlet Allocation).
    
    :param text: Raw transcription text to model topics on.
    :param num_topics: The number of topics to extract.
    :return: A dictionary of topics with top words.
    """
    try:
        # Preprocess the text with spaCy: lemmatize and remove stopwords and punctuation
        doc = nlp(text)
        cleaned_text = ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

        # Vectorize the cleaned text
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform([cleaned_text])

        # Fit the LDA model to the vectorized text
        lda = LDA(n_components=num_topics, random_state=42)
        lda.fit(X)

        # Get the topic distribution for the document
        topic_distribution = lda.transform(X)  # Proportion of each topic in the document

        # Get the topics and words (top words for each topic)
        topics = lda.components_
        feature_names = vectorizer.get_feature_names_out()
        topic_words = {}

        for topic_idx, topic in enumerate(topics):
            top_words = [feature_names[i] for i in topic.argsort()[:-6 - 1:-1]]  # Get top 5 words for each topic
            topic_words[topic_idx] = top_words

        # Output the topic distribution for the document and top words per topic
        print("Topic Distribution in Document:")
        for idx, doc_topic_dist in enumerate(topic_distribution):
            print(f"\nDocument {idx + 1} topic distribution:")
            for topic_idx, dist in enumerate(doc_topic_dist):
                print(f"  Topic {topic_idx + 1}: {dist * 100:.2f}%")
        
        print("\nTop words per topic:")
        for topic_idx, words in topic_words.items():
            print(f"  Topic {topic_idx + 1}: {', '.join(words)}")
        
        return topic_words

    except Exception as e:
        print(f"Error during topic modeling: {e}")
        return None

