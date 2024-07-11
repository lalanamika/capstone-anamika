# Streamlit app for SmartRecipes - a recipe recommender system that uses a
# model trained on 20000 recipes to recommend recipes that contain
# ingredients that a user has.
# ===================================================

# ===================================================
# import libraries
# NOTE - I had to manually do pip install nltk in the conda environment - for the spell checker
# NOTE - I had to manually do pip install pattern in the conda environment - for the pluralizer
# ===================================================
import pandas as pd
import streamlit as st
import numpy as np

import joblib

# for spell checker
import nltk
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams

from pattern.en import pluralize
from pattern.en import singularize
# Installing NLTK data to import
# and run en module of pattern
# NOTE - This needs to be done the first time this is run.
# nltk.download('popular')

from sys import path
path.append("../notebooks/")
import cust_tokenizer

# ===================================================
# Setup the web page
# ===================================================
st.set_page_config("SmartRecipes", ":green_salad:", layout="wide")

# Custom HTML/CSS for the banner - for this the image has to be on a URL, not on local machine
custom_html = """
<div class="banner">

    <img src="https://images.unsplash.com/photo-1463123081488-789f998ac9c4?q=80&w=1470&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="Banner Image">
</div>
<div class="banner-text">
    <h1>SmartRecipes</h1>
</div>
<style>
    .banner {
        width: 160%;
        height: 200px;
        overflow: hidden;
    }
    .banner img {
        width: 100%;
        object-fit: cover;
    }
    .banner-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: left;
        z-index: 1;
        color: #fff;
    }

    .banner-text h1 {
        font-size: 5rem;
        margin-bottom: 20px;
    }
</style>
"""
# Display the custom HTML
st.components.v1.html(custom_html)

st.subheader(":green_salad: Let's see some recipes! :green_salad:")

# ===================================================
# Define the relevant functions
# ===================================================
@st.cache_data # <- add decorators after tried running the load multiple times
def load_data(path_to_file):
    """
    This function loads the csv into a Pandas DataFrame

    input parameters:
        path - string - path to the csv file

    returns:
        a Pandas DataFrame object
    """
    df = pd.read_csv(path_to_file)
    return df


def spellcheck(seriesOfWords, vocab, writeToScreen):
    """
    This functions does a spell check using jaccard_distance, we set a threshold of 0.5.
    If the distance is < 0.5, the word will be corrected, otherwise an error message will be
    sent to the user.
    NOTE: If there is a 2 word ingredient, like "peanut butter", then this will not spell check it.

    input parameters:
        seriesOfWords - list of words
        vocab - list containing the vocabulary of the trained CountVectorizer
        writeToScreen - boolean. Set to True if you want the error message to be printed to
                        the web page for the user to see. Set to False if not.

    returns:
        list of corrected words.
    """
    final_words = []
    for word in seriesOfWords:
        # if the word contains space, it is a 2 word ingredient, we won't spell check
        if " " in word:
            final_words.append(word)
        else:
            # convert to lowercase and remove any whitespace
            word1 = word.lower()
            word1_ns = word1.strip()
            temp = [(jaccard_distance(set(ngrams(word1_ns, 2)),
                                    set(ngrams(w, 2))),w)
                    for w in vocab if w[0]==word1_ns[0]]
            sorted_temp = sorted(temp, key = lambda val:val[0])
            word_distance = sorted_temp[0][0]
            corrected_word = sorted_temp[0][1]
            if word_distance > 0.5:
                if writeToScreen:
                    st.write(f":red[Sorry we could not recognize the ingredient {word}]")
            else:
                final_words.append(corrected_word)
    # st.write(final_words)
    return final_words


def addPluralsAndSingulars(seriesOfWords, vocab):
    """
    This function will pluralize / singularize the words.
    NOTE: If there is a 2 word ingredient, like "peanut butter", this will not
    pluralize it.

    input parameters:
        seriesOfWords - a list of words
        vocab - a list containing the vocabulary of the trained CountVectorizer

    returns:
         a list containing the singular / plural from for the words
    """
    res1 = []
    for word in seriesOfWords:
        if " " in word:
            pass
        else:
            res1.append(pluralize(word))
            res1.append(singularize(word))
    if len(res1) > 0:
        spell_checked = spellcheck(set(res1), vocab, False)
        return spell_checked
    else:
        return res1


def processInputString(input_string, vocab):
    """
    This function splits the input string (comma separated or space separated) into a list,
    runs a spell check on the words, and also adds the plural / singular form of the words
    to the result.
    NOTE: If there is a 2 word ingredient, like "peanut butter", then you need commas to
    specify other ingredients

    input parameters:
        input_string - comma or space separated string, containing words entered by the user
        vocab - a list containing the vocabulary of the trained CountVectorizer
    returns:
        a list of words that includes the corrected words for any spelling mistakes, and their
        singular / plural form too.
    """
    result_list = []
    processed_string = ""
    if input_string is not None and input_string != "" and input_string != " ":
        separator = " "
        if "," in input_string:
            separator = ","

        ing_list = input_string.split(separator)
        corrected_ing_list = spellcheck(ing_list, vocab, True)
        result_list.extend(corrected_ing_list)
        plurals_list = addPluralsAndSingulars(corrected_ing_list, vocab)
        for ing in plurals_list:
            if ing not in result_list:
                result_list.append(ing)
        processed_string = " ".join(result_list)
    return processed_string


# ===================================================
# Load the data, trained vectorizer and model
# ===================================================
df = load_data("../data/final/full_recipes.csv")
model = joblib.load('../model/model_final.pkl')
new_vocab_list = joblib.load('../model/custom_vocab.pkl')
vect = joblib.load('../model/vect_mod.pkl')

# Add a radio button for categories
genre = st.radio(
    "Are you looking for any specific category of recipe?",
    ["No", ":green[Vegetarian] :leafy_green:", "Gluten-Free", "Dessert :cake:"])

category_string = ""
if genre == ":green[Vegetarian] :leafy_green:":
    category_string = "Vegetarian"
elif genre == "Gluten-Free":
    category_string = "Gluten free"
elif genre == "Dessert :cake:":
    category_string = "Dessert"

# Add text boxes for "include ingredients" and "exclude ingredients"
yes_ing = st.text_input('Which ingredients do you want to use?', 'Chicken, Potato')
no_ing = st.text_input('If there is an ingredient you want to avoid, enter it below.', '')

# ==============================================================
# Process the user input to make it ready for the model
# Spell check and pluralize / singularize the input ingredients
# Also combine the category choice with the "include ingredients"
# ==============================================================
include_ing_string = category_string + " " + processInputString(yes_ing, new_vocab_list)
exclude_ing_string = processInputString(no_ing, new_vocab_list)

# Transform the processed version of the "include ingredients" and "exclude ingredients"
# to come up with a vector to feed to the trained model.
updated_ing_tx = None
yes_ing_tx = None
no_ing_tx = None

if include_ing_string is not None and include_ing_string != "" and include_ing_string != " ":
    yes_ing_series = pd.Series(include_ing_string)
    yes_ing_tx = vect.transform(yes_ing_series)

if exclude_ing_string is not None and exclude_ing_string != "" and exclude_ing_string != " ":
    no_ing_series = pd.Series(exclude_ing_string)
    no_ing_tx = (vect.transform(no_ing_series)) * -1

if yes_ing_tx is not None:
    if no_ing_tx is not None:
        updated_ing_tx = yes_ing_tx + no_ing_tx
    else: # no ing is none
        updated_ing_tx = yes_ing_tx
else:
    if no_ing_tx is not None:
        updated_ing_tx = no_ing_tx

# ==============================================================
# Run the model if there is some valid user input
# ==============================================================
if updated_ing_tx is not None:
    distances, indices = model.kneighbors(updated_ing_tx)

    # Show the result on the web page
    for i in range(0, 11):  # NOTE: 11 can be made configurable and match the n-neighbors number
        name = df.loc[indices[0][i], ['title']].values[0]
        ingredients = df.loc[indices[0][i], ['ingredientsStr']].values[0].split("',")
        steps = df.loc[indices[0][i], ['directionsStr']].values[0]
        with st.expander(name):
            st.write(":blue[Ingredients]")
            for food in ingredients:
                st.write(food)
            st.write(":blue[Steps]")
            st.write(steps)


# ==============================================================
# Test Cases
# ==============================================================
# Add only beetroot for include ing - No recipes shown only error message
# Add only category with no ingredients in inc and exc
# Exclude something with nothing in inclusion
# Test 2 word ingredients - Peanut butter
# Test spell checker - Peanut butter, brocoli
# Test pluralization - Chicken Potato
# ==============================================================