# Import packages
import pandas as pd 
import nltk
from nltk import word_tokenize
from nltk import FreqDist

# Check gutenberg fileids to select a text
nltk.corpus.gutenberg.fileids()

# Assign The Man Who Was Thursday by G.K. Chesterton to file
file = 'chesterton-thursday.txt'

# Import file; check type, length, and first 120 words.
thursday = nltk.corpus.gutenberg.raw(file)

type(thursday)
len(thursday)
thursday[:120]

# Tokenize thursday 
thursday_tokens = word_tokenize(thursday)
len(thursday_tokens)
thursday_tokens[:50]

# Lowercase all token words 
thursday_words = [w.lower() for w in thursday_tokens]
thursday_words[:50]

# Sort words and build vocabulary 
thursday_vocab = sorted(set(thursday_words))
thursday_vocab[:50]

# How many times to the words sunday, thursday, and mob apper?
thursday_words.count('sunday')   #97
thursday_words.count('thursday') #21
thursday_words.count('mob')      # 9

# Create a dictionary with the words as keys and the number of times they occur as the values
fdist = FreqDist(thursday_words)
fdist_keys = list(fdist.keys())

fdist['thursday']

# Look at the most frequent words
top_keys = fdist.most_common(30)

# Print 30 most common pairs
for pair in top_keys:
    print(pair)

# Count the number of words to normalize the word distribution
num_words = len(thursday_words)
top_keys_norm = [(word, freq/num_words) for (word, freq) in top_keys]

# Print 30 most common normalized pairs
for pair in top_keys_norm:
    print(pair)