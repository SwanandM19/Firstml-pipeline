
import nltk
nltk.download('punkt')  # Tokenization (required)
nltk.download('stopwords')  # Common stopwords
nltk.download('wordnet')  # Lemmatization
nltk.download('averaged_perceptron_tagger')  # POS tagging (if needed)

from nltk.tokenize import word_tokenize

text = "This is a test sentence."
print(word_tokenize(text, language='english'))