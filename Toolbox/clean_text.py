# a bunch of code line to preprocess text (punctuation;lowercase;
# tokenize,stopwords,remove numbers)
# bag of words modeling and N_gram modeling
# tune a vectorizer

"""# when installing nltk for the first time we need to also download a few built in libraries
!pip install nltk
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
"""


# a function in order to clean the text

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import unidecode


def clean (text):

    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ') # Remove Punctuation

    lowercased = text.lower() # Lower Case

    unaccented_string = unidecode.unidecode(lowercased) # remove accents

    tokenized = word_tokenize(unaccented_string) # Tokenize

    words_only = [word for word in tokenized if word.isalpha()] # Remove numbers

    stop_words = set(stopwords.words('portuguese')) # Make stopword list

    without_stopwords = [word for word in words_only if not word in stop_words] # Remove Stop Words

    return " ".join(without_stopwords)

#df['clean_text'] = df['title_comment'].apply(clean)


# a function that remove punctuation
import string

string.punctuation

def remove_punctuation (line):
    for punctuation in string.punctuation:
        line = line.replace(punctuation, "")
    return line

# apply the remove_punctuation function
#df['clean_text'] = df['text'].apply(remove_punctuation)


# a function that remove number
def remove_numbers (text):
    return ''.join(word for word in text if not word.isdigit())

#apply the remove number to the clan text
# df['clean_text] = df['clean_text].apply(remove_numer)

# a function that remove StopWords

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
def remove_StopWords (text):
    word_tokens = word_tokenize(text)
    return ' '.join( [w for w in word_tokens if not w in stop_words])

#df['clean_text'] = df['text_clean'].apply(remove_StopWords)

#df.head()

# a function to lemmatize the text
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize (text):
    return ' '.join ([lemmatizer.lemmatize(word) for word in text.split()])

#df['clean text'] = df['clean_text'].apply(lemmatize)



# a function to remove emails address from a text
import re

def remove_emails(df, label):
    """ This function removes email adresses
        inputs:
         - text """
    df[label] = df[label].apply(lambda x: re.sub(
        r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])""",
        " ", x))
    return df

# Bag-of_words Modelling
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(df['clean_text'])
X_bow.toarray()

vectorizer.get_feature_names()

dir (vectorizer)

# cross validate the performance of the model
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df.spam

nb_model = MultinomialNB()
result = cross_validate (nb_model,X,y,cv=3)
result['test_score'].mean()

"""
other sample of prepprocessing for text analysis
"""
# first do the import
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

# make a function to remove puntuation and lower case the text
def preprocessing(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')
    lowercased = text.lower()
    return lowercased

# apply the preprocessing function to the data

data['clean_reviews'] = data.reviews.apply(preprocessing)

"""
👇 Using `cross_validate`, score a Multinomial Naive Bayes model trained
on a Bag-of-Word representation of the texts.
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate

vectorizer = CountVectorizer()

X_bow = vectorizer.fit_transform(data.clean_reviews)

cv_nb = cross_validate(MultinomialNB(), X_bow, data.target, scoring="accuracy")

cv_nb['test_score'].mean()

"""
 Using `cross_validate`, score a Multinomial Naive Bayes model
 trained on a 2-gram Bag-of-Word representation of the texts.
 """
vectorizer = CountVectorizer(ngram_range=(2,2))

X_ngram = vectorizer.fit_transform(data.clean_reviews)

cv_nb = cross_validate( MultinomialNB(), X_ngram, data.target, scoring = "accuracy")

cv_nb['test_score'].mean()




'''
Tune a vectorizer of your choice
(or try both!) and a MultinomialNB model simultaneously.
'''

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Create Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB()),
])

# Set parameters to search
parameters = {
    'tfidf__ngram_range': ((1, 1), (2, 2)),
    'tfidf__min_df': (0.05, 0.1),
    'tfidf__max_df': (0.75, 1),
    'nb__alpha': (0.01, 0.1, 1, 10),
}

# Perform grid search on pipeline
grid_search = GridSearchCV(pipeline,
                           parameters,
                           n_jobs=-1,
                           verbose=1,
                           scoring="accuracy",
                           refit=True,
                           cv=5)

grid_search.fit(data.clean_reviews, data.target)
