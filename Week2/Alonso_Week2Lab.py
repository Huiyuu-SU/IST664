#%% [markdown]
## IST664 Week 2 Lab 
# 2019-04-13

#%% 
# Load librares 
import nltk 
from nltk import FreqDist
from nltk import word_tokenize
from nltk import bigrams
from nltk.collocations import *
from nltk.corpus import stopwords
import re

#%% [markdown]
# Choose a file from the Gutenberg corpus, and build a bigram finder.  
# Apply filters and run raw frequency and pmi scorers to compare results.  

#%% 
file = 'chesterton-thursday.txt'
thursday = nltk.corpus.gutenberg.raw(file)

#%% [markdown]
# Steps to take place:
# 
#1. tokenize  
#2. lowercase  
#3. create bigram finder and scorer  
#4. apply re filter  
#5. remove stopwords  
#6. remove low frequency words  
#7. repeat steps 4-6 with pmi  

#%%
# Tokenize thursday 
thursday_tokens = word_tokenize(thursday)
print(len(thursday_tokens))

# Lowercase all token words 
thursday_words = [w.lower() for w in thursday_tokens]
thursday_words[:50]

#%% [markdown]
# Now that the text has been lowercased and tokenized, we'll start working through
# the different bigram scorers. 

#%%
bigram_measures = nltk.collocations.BigramAssocMeasures()

# Create the bigram finder and score bigram frequency
finder = BigramCollocationFinder.from_words(thursday_words)
scored = finder.score_ngrams(bigram_measures.raw_freq)

# scored is a list of bigram pairs with their score
print(type(scored))
first = scored[0]
print(type(first))
print(first)

#%%
# scores are sorted in decreasing frequency
for bscore in scored[:30]:
    print (bscore)


#%%
# function that takes a word and returns true if it consists only
#   of non-alphabetic characters  (assumes import re)
def alpha_filter(w):
  # pattern to match word of non-alphabetical characters
  pattern = re.compile('^[^a-z]+$')
  if (pattern.match(w)):
    return True
  else:
    return False

#%%
# apply a filter to remove non-alphabetical tokens from the emma bigram finder
finder.apply_word_filter(alpha_filter)
scored = finder.score_ngrams(bigram_measures.raw_freq)
for bscore in scored[:30]:
    print (bscore)

#%%
# apply a filter to remove stop words
stopwords = stopwords.words("english")
finder.apply_word_filter(lambda w: w in stopwords)
scored = finder.score_ngrams(bigram_measures.raw_freq)
for bscore in scored[:20]:
    print (bscore)

#%%
# apply a filter (on a new finder) to remove low frequency words
finder.apply_freq_filter(5)
scored = finder.score_ngrams(bigram_measures.raw_freq)
for bscore in scored[:20]:
    print (bscore)

#%%
### pointwise mutual information
finder3 = BigramCollocationFinder.from_words(thursday_words)
scored = finder3.score_ngrams(bigram_measures.pmi)
for bscore in scored[:20]:
    print (bscore)

#%%

# to get good results, must first apply frequency filter
finder.apply_freq_filter(5)
scored = finder.score_ngrams(bigram_measures.pmi)
for bscore in scored[:30]:
    print (bscore)

#%%
