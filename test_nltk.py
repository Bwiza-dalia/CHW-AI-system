import nltk
nltk.data.path.append('/Users/a/nltk_data')
from nltk.tokenize import word_tokenize

print(word_tokenize("This is a test."))