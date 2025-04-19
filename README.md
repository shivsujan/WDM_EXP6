### EX6 Information Retrieval Using Vector Space Model in Python
### DATE:19.04.2025 
### AIM: To implement Information Retrieval Using Vector Space Model in Python.
### Description: 
<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

### Program:

    import requests
    from bs4 import BeautifulSoup
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import string
    import nltk

    # Download the necessary NLTK resources
    nltk.download('punkt', force=True)
    nltk.download('stopwords', force=True)
    nltk.download('punkt_tab')

###### Sample documents stored in a dictionary
    documents = {
    "doc1": "The Cat sat on the mat",
    "doc2": "The Dog sat on the table",
    "doc3": "The Cat lay on the rug",
    }
###### Preprocessing function to tokenize and remove stopwords/punctuation
    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token not in stopwords.words("english") and token not in string.punctuation]
        return " ".join(tokens)

###### Preprocess documents and store them in a dictionary
    preprocessed_docs = {doc_id: preprocess_text(doc) for doc_id, doc in documents.items()}

###### Construct TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs.values())

###### Calculate cosine similarity between query and documents
    def search(query, tfidf_matrix, tfidf_vectorizer):
    query_processed = preprocess_text(query)
    query_vector = tfidf_vectorizer.transform([query_processed])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Sort documents by similarity score
    sorted_indexes = similarity_scores.argsort()[::-1]

###### Return sorted documents along with their similarity scores
    results = []
    for idx in sorted_indexes:
        doc_id = list(preprocessed_docs.keys())[idx]
        score = similarity_scores[idx]
        results.append((doc_id, documents[doc_id], score))
    return results
###### Get input from user
    query = input("Enter your query: ")

###### Perform search
    search_results = search(query, tfidf_matrix, tfidf_vectorizer)

###### Display search results
    print("Query:", query)
    for i, result in enumerate(search_results, start=1):
    print(f"\nRank: {i}")
    print("Document ID:", result[0])
    print("Document:", result[1])
    print("Similarity Score:", result[2])
    print("----------------------")

###### Get the highest rank cosine score
    highest_rank_score = max(result[2] for result in search_results)
    print("\nThe highest rank cosine score is:", highest_rank_score)

### Output:
![Screenshot 2025-04-19 135022](https://github.com/user-attachments/assets/2d0ca198-769c-4434-8014-63f97d3c3626)


### Result:
Thus the implementation Information Retrieval Using Vector Space Model in Python is successfullly executed.
### EX6 Information Retrieval Using Vector Space Model in Python
### DATE:19.04.2025 
### AIM: To implement Information Retrieval Using Vector Space Model in Python.
### Description: 
<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

### Program:

    import requests
    from bs4 import BeautifulSoup
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import string
    import nltk

    # Download the necessary NLTK resources
    nltk.download('punkt', force=True)
    nltk.download('stopwords', force=True)
    nltk.download('punkt_tab')

###### Sample documents stored in a dictionary
    documents = {
    "doc1": "The Cat sat on the mat",
    "doc2": "The Dog sat on the table",
    "doc3": "The Cat lay on the rug",
    }
###### Preprocessing function to tokenize and remove stopwords/punctuation
    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token not in stopwords.words("english") and token not in string.punctuation]
        return " ".join(tokens)

###### Preprocess documents and store them in a dictionary
    preprocessed_docs = {doc_id: preprocess_text(doc) for doc_id, doc in documents.items()}

###### Construct TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs.values())

###### Calculate cosine similarity between query and documents
    def search(query, tfidf_matrix, tfidf_vectorizer):
    query_processed = preprocess_text(query)
    query_vector = tfidf_vectorizer.transform([query_processed])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Sort documents by similarity score
    sorted_indexes = similarity_scores.argsort()[::-1]

###### Return sorted documents along with their similarity scores
    results = []
    for idx in sorted_indexes:
        doc_id = list(preprocessed_docs.keys())[idx]
        score = similarity_scores[idx]
        results.append((doc_id, documents[doc_id], score))
    return results
###### Get input from user
    query = input("Enter your query: ")

###### Perform search
    search_results = search(query, tfidf_matrix, tfidf_vectorizer)

###### Display search results
    print("Query:", query)
    for i, result in enumerate(search_results, start=1):
    print(f"\nRank: {i}")
    print("Document ID:", result[0])
    print("Document:", result[1])
    print("Similarity Score:", result[2])
    print("----------------------")

###### Get the highest rank cosine score
    highest_rank_score = max(result[2] for result in search_results)
    print("\nThe highest rank cosine score is:", highest_rank_score)

### Output:
![Screenshot 2025-04-19 135022](https://github.com/user-attachments/assets/2d0ca198-769c-4434-8014-63f97d3c3626)


### Result:
Thus the implementation Information Retrieval Using Vector Space Model in Python is successfullly executed.
