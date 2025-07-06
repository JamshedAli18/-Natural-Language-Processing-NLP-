import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords and tokenizer
nltk.download('stopwords')
nltk.download('punkt')

text = "This is a simple example to show how to remove stopwords from text."

# Tokenize the text
words = word_tokenize(text)

# Get English stopwords
stop_words = set(stopwords.words('english'))

# Remove stopwords
filtered_words = [word for word in words if word.lower() not in stop_words]

print("Original:", words)
print("Without Stopwords:", filtered_words)
