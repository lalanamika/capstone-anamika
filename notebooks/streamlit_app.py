### import libraries
# NOTE - I had to manually do pip install nltk in the conda environment - for the spell checker
# NOTE - I had to manually do pip install pattern in the conda environment - for the pluralizer

import pandas as pd
import streamlit as st
import numpy as np

# for spell checker
import nltk
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams

from pattern.en import pluralize
from pattern.en import singularize
# Installing NLTK data to import
# and run en module of pattern

# CHECK - IS IT OK TO RUN THIS JUST THE FIRST TIME? OR DO I NEED TO RUN IT ON EVERY REBOOT?
# nltk.download('popular')


# TODO:
# Add support for Peanut butter back
# If input list is empty , dont run model
# add support for comma separated?
import joblib
import cust_tokenizer

st.set_page_config("SmartRecipes", ":green_salad:", layout="wide")

# Custom HTML/CSS for the banner - for this the image has to be on a URL, not on local machine
custom_html = """
<div class="banner">
    <img src="https://cdn.stocksnap.io/img-thumbs/960w/peppers-vegetables_CSIVDF12OA.jpg" alt="Banner Image">
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
        text-align: center;
        z-index: 1;
        color: #fff;
    }

    .banner-text h1 {
        font-size: 3rem;
        margin-bottom: 20px;
    }
</style>
"""
# Display the custom HTML
st.components.v1.html(custom_html)

### To position text and color, you can use html syntax
# st.markdown("<h1 style='text-align: center; color: darkblue;'>SmartRecipes</h1>", unsafe_allow_html=True)

# ===================================================
# Load the dataset
@st.cache_data # <- add decorators after tried running the load multiple times
def load_data(path):

    df = df = pd.read_csv(path)
    return df

def spellcheck(seriesOfWords, vocab, writeToScreen):
    # TODO : Add docstring here
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
                st.write(f":red[Sorry we could not recognize the ingredient {word}]")
            else:
                final_words.append(corrected_word)
    # st.write(final_words)
    return final_words

def addPluralsAndSingulars(seriesOfWords, vocab):
    # TODO : Add docstring here
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

# ================================================
df = load_data("../data/final/full_recipes.csv")

st.subheader("Let's see some recipes!")

include_ing_list = []
exclude_ing_list = []

genre = st.radio(
    "Are you looking for any specific category of recipe?",
    ["No", ":green[Vegetarian] :leafy_green:", "Gluten-Free", "Dessert :cake:"])

if genre == ":green[Vegetarian] :leafy_green:":
    include_ing_list.append("Vegetarian")
elif genre == "Gluten-Free":
    include_ing_list.append("Gluten free")
elif genre == "Dessert :cake:":
    include_ing_list.append("Dessert")


# A. Load the model using joblib
model = joblib.load('model_final.pkl')
new_vocab_list = joblib.load('custom_vocab.pkl')
vect = joblib.load('vect_mod.pkl')

# B - Use desired ingredients and undesired ingredients
yes_ing = st.text_input('Enter the ingredients to include below, separated by commas', 'Chicken, Potato')
no_ing = st.text_input('Enter the ingredients to exclude below, separated by commas', 'None')

# figure out separator
separator = " "
if "," in yes_ing:
    separator = ","

yes_ing_list = yes_ing.split(separator)
corrected_ing_list_inc = spellcheck(yes_ing_list, new_vocab_list, True)
include_ing_list.extend(corrected_ing_list_inc)
plurals_list_inc = addPluralsAndSingulars(corrected_ing_list_inc, new_vocab_list)
for ing in plurals_list_inc:
    if ing not in include_ing_list:
        include_ing_list.append(ing)
includeIngString = " ".join(include_ing_list)
# st.write(includeIngString)

yes_ing_series = pd.Series(includeIngString)
# st.write(corrected_ing_list)

if no_ing != "None" and no_ing != "":
    no_ing_list = no_ing.split(separator)
    corrected_ing_list_exc = spellcheck(no_ing_list, new_vocab_list, True)
    exclude_ing_list.extend(corrected_ing_list_exc)
    plurals_list_exc = addPluralsAndSingulars(corrected_ing_list_exc, new_vocab_list)
    for ing in plurals_list_exc:
        if ing not in exclude_ing_list:
            exclude_ing_list.append(ing)
    excludeIngString = " ".join(exclude_ing_list)
    # st.write(excludeIngString)

yes_ing_series = pd.Series(includeIngString)
yes_ing_tx = vect.transform(yes_ing_series)

updated_ing_tx = None
if len(exclude_ing_list) > 0:
    no_ing_series = pd.Series(no_ing)
    no_ing_tx = (vect.transform(no_ing_series)) * -1
    updated_ing_tx = yes_ing_tx + no_ing_tx
else:
    updated_ing_tx = yes_ing_tx

distances, indices = model.kneighbors(updated_ing_tx)

# C.  Print result
for i in range(0, 11):  # TODO: 11 should be made configurable and match the n-neighbors number
    name = df.loc[indices[0][i], ['title']].values[0]
    distance = (distances[0][i]).round(3)
    rating = df.loc[indices[0][i], ['rating']].values[0]
    ingredients = df.loc[indices[0][i], ['ingredientsStr']].values[0].split("',")
    steps = df.loc[indices[0][i], ['directionsStr']].values[0]
    # st.write(f"{name}  :  {distance}  :  {rating}")
    with st.expander(name):
        st.write(ingredients)
        st.write(steps)
