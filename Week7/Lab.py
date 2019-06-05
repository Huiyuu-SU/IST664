#%%   
import nltk

#%% 
# Define a feature extraction function for each name 
def gender_features3(word): 
    return{"suffix1": word[-1], "suffix2": word[-2]}

#%%
print(gender_features3("Shrek"))

#%%
from nltk.corpus import names 

#%%
namesgender = ([(name, "male") for name in names.words("male.txt")] + [(name, "female") for name in names.words("female.txt")])
print(len(namesgender))
print(namesgender[:20])
print(namesgender[7924:])

#%%
import random 
random.shuffle(namesgender)
print(namesgender[:20])

#%%
# Separate into training and testing set 
train_names = namesgender[:6355]
test_names = namesgender[6355:]

#%%
# use features to train a classifier and test on a development test set 
train_set = [(gender_features3(n), g) for (n, g) in train_names]
test_set = [(gender_features3(n), g) for (n, g) in test_names]
print(train_set[:20])

#%%
classifier = nltk.NaiveBayesClassifier.train(train_set)

#%%
print(classifier.classify(gender_features3("Luke")))
print(classifier.classify(gender_features3("Leia")))

#%%
print(nltk.classify.accuracy(classifier, test_set))

#%%
print(classifier.show_most_informative_features(20))

#%%
# Define a function that will compare the classifier labels with the gold standard labels 
def geterrors(test):
    errors = []
    for (name, tag) in test: 
        guess = classifier.classify(gender_features3(name))
        if guess != tag:
            errors.append((tag, guess, name))
    return errors

#%%
errors = geterrors(test_names)
print(len(errors))

#%%
def printerrors(errors): 
    for (tag, guess, name) in sorted(errors): 
        print("correct={:<8s} guess={:<8s} name={:<30s}".format(tag, guess, name))

#%%
printerrors(errors)

#%%
from nltk.corpus import brown

#%%
# define features for the i-th word in the sentence, including three types of 
# suffix and one pre-word. 
# the pos features function takes the sentence of untagged words and the index 
# of a word i. it creates features for word i, including the word i-1 
def pos_features(sentence, i):
    features = {"suffix(1)": sentence[i][-1:], 
                "suffix(2)": sentence[i][-2:],
                "suffix(3)": sentence[i][-3:]}
    if i == 0: 
        features["prev-word"] = "<START>"
    else: 
        features["prev-word"] = sentence[i-1]
    return features

#%%
sentence0 = brown.sents()[0]
print(sentence0)
# word 8 of sentence 0
print(sentence0[8])

#%%
pos_features(sentence0, 8)

#%%
# Get the POS tagged sentences with categories of news
tagged_sents = brown.tagged_sents(categories="news")
tag_sent0 = tagged_sents[0]
tag_sent0

#%%
nltk.tag.untag(tag_sent0)

#%%
# the python enumerate function generates an index number for each item in the list
for i, (word, tag) in enumerate(tag_sent0):
    print(i, word, tag)

#%%
# get features sets of words appearing in the corpus, from untagged sentences
# and then get their tags from corresponding tagged sentences.
# use the python function enumerate to pair the index numbers with sentence words
# for the pos features function 
featuresets = []
for tagged_sent in tagged_sents: 
    untagged_sent = nltk.tag.untag(tagged_sent)
    for i, (word, tag) in enumerate(tagged_sent):
        featuresets.append((pos_features(untagged_sent, i), tag))

#%%
for f in featuresets[:10]: 
    print(f)

#%%
# using Naive Bayes as a classifier: split the data into a training and testing set.
# use a 90/10 split 
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
print(len(train_set))
print(len(test_set))

#%%
# train the classifier on the training set 
classifier = nltk.NaiveBayesClassifier.train(train_set)

#%%
# evaluate the accuracy of the classifier 
print(nltk.classify.accuracy(classifier, test_set))

#%%
from nltk.corpus import movie_reviews

#%%
print(movie_reviews.categories())

#%%
documents = [(list(movie_reviews.words(fileid)), category) 
                for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)]
print(len(documents))

#%%
random.shuffle(documents)
print(documents[0])

#%%
# use words from all documents to define the word vector for features 
# get all words from all movie_reviews and put into a frequency distribution 
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
print(len(all_words))

#%%
# get the 2000 most frequent words from the corpus 
word_items = all_words.most_common(5000)
word_features = [word for (word, freq) in word_items]

#%%
print(word_features[:100])

#%%
# define features of a document
# each feature is 'contains(keyword)' and is true or false depending on whether
# that keyword is in the document 
def document_features(document, word_features): 
    document_words = set(document)
    features = {}
    for word in word_features: 
        features["V_%s" % word] = (word in document_words)
    return features 

#%%
# get feature sets for a document, including keyword features and category feature 
featuresets = [(document_features(d, word_features), c) for (d,c) in documents]

#%%
print(featuresets[0])

#%%
# train using Naive Bayes classifier with 95/5 split 
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set) 

#%%
# evaluate the accuracy of the classifier 
print(nltk.classify.accuracy(classifier, test_set))

#%%
# Show which features of classifier are most informative 
print(classifier.show_most_informative_features(30))

#%%
