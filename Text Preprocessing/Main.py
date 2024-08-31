import spacy 
from collections import Counter
from nltk.stem import PorterStemmer

nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

text = """
Python is a high-level, general-purpose programming language.
Its design philosophy emphasizes code readability with the use of significant indentation.
Python is dynamically typed and garbage-collected. It supports multiple programming paradigms,
including structured (particularly procedural), object-oriented and functional programming.
"""

doc = nlp(text)

# 1- Lower casing
text_lowercased = " ".join(token.text.lower() for token in doc)
# print(text_lowercased)

# 2- Removal of Punctuations
text_no_punctuation = " ".join(token.text for token in doc if not token.is_punct)
# print(text_no_punctuation)

# 3- Removal of Stopwords
text_no_stopwords = " ".join(token.text for token in doc if not token.is_stop)
# print(text_no_stopwords)

# 4- Removal of Frequent words
stop_words = spacy.lang.en.stop_words.STOP_WORDS
non_stop_words = [token.text.lower() for token in doc if token.text.lower() not in stop_words]
word_frequencies = Counter(non_stop_words)
frequency_threshold = 2
words_to_remove = [word for word, frequency in word_frequencies.items() if frequency >= frequency_threshold]
text_no_frequent_words = " ".join(token.text for token in doc if token.text.lower() not in words_to_remove)
# print(text_no_frequent_words)

# 5- Removal of Rare words
all_words = [token.text.lower() for token in doc]
word_frequencies = Counter(all_words)
frequency_threshold = 1
words_to_remove = [word for word, frequency in word_frequencies.items() if frequency <= frequency_threshold]
text_no_rare_words = " ".join(token.text for token in doc if token.text.lower() not in words_to_remove)
# print(text_no_rare_words)

# 6- Stemming
text_after_stemming = " ".join(stemmer.stem(token.text) for token in doc)
# print(text_after_stemming)

# 7- Lemmatization
text_after_lemmatization = " ".join(token.lemma_ for token in doc)
# print(text_after_lemmatization)
