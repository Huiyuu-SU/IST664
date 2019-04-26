# Import required packages
import docx2txt
import nltk
from nltk import FreqDist
from nltk import word_tokenize 
import os
from io import StringIO

# Change directory and load the document
os.chdir("C:/Users/malon/Documents/Syracuse University/IST 664 Natural Language Processing/Week2")
infile = 'nlp_lecture_exercise_bigram_predictive_probabilities.docx'

# Read document 
text = docx2txt.process(infile)

# Convert document to tokens, retain tokens after '2013:' 
text_tokens = word_tokenize(text)
speech_of_interest = text_tokens[23:]

# Initiate empty list to store lowercased words. 
speech_lower = []

for word in speech_of_interest: 
    word = word.lower()
    speech_lower.append(word)

# Create a dictionary with the lowercased words
word_distr = FreqDist(speech_lower)

word_distr['we']
word_distr['believe']
word_distr['our']