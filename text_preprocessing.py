# text_preprocessing.py

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from spellchecker import SpellChecker
import Levenshtein


# Function to ensure the necessary NLTK data is available
def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
        
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

# Ensure the necessary NLTK data is downloaded
ensure_nltk_data()
        
def get_wordnet_pos(treebank_tag):
    """Map treebank part-of-speech tags to WordNet part-of-speech tags."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun


new_words = ['const','road work','work','roadwork','constru','const','road constru',
            'road const','wz','flagger','flag','const access','road wk','wk','workzone','work zone',
            'rd work','rd wk','rdwk', 'nov','apr','jul','aug']


#add words to the vocab of the spellchecker and spellcheck
spell = SpellChecker()   
spell.word_frequency.load_words(new_words) 
def correct_spelling(text):
    words = text.split()
    corrected_words = [spell.correction(word) if spell.correction(word) is not None else word for word in words]
    return ' '.join(corrected_words)

#Levenshtein
# 1 for max distance to handle double typing of letters otherwise if =2 is transform into wz 
def correct_Levenshtein(text, keywords=['wz'], max_distance=1):
    corrected_words = []
    words = text.split()
    for word in words:
        replacement_word = word  # Default to the original word
        for keyword in keywords:
            distance = Levenshtein.distance(word, keyword)
            if distance <= max_distance:
                replacement_word = keyword  # Replace with the correct keyword
                break  # No need to check other keywords
        corrected_words.append(replacement_word)
    return ' '.join(corrected_words)



def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    tokenized_words = word_tokenize(sentence)
    pos_tagged = nltk.pos_tag(tokenized_words)
    lemmatized_words = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(pos)) for word, pos in pos_tagged]
    return ' '.join(lemmatized_words)

#if i want to use multiple preprocessing in model_loading.py
#create the pipeline to use
#for example if i want levenshtein and spellchecker (which i call directly in app.py)
# def preprocess_pipeline_1(text, keywords):
#     corrected_text = correct_spelling(text)
#     further_corrected_text = correct_Levenshtein(corrected_text, keywords)
#     return further_corrected_text