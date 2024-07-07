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
                if writeToScreen:
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

def processInputString(input_string, new_vocab_list):
    # TODO: Add docstring here
    # Figure out separator, users can enter comma separated or space separated list of ingredients
    # NOTE: If there is a 2 word ingredient, like "peanut butter", then you need commas to specify other ingredients
    result_list = []
    processed_string = ""
    if input_string is not None and input_string != "" and input_string != " ":
        separator = " "
        if "," in input_string:
            separator = ","

        ing_list = input_string.split(separator)
        corrected_ing_list = spellcheck(ing_list, new_vocab_list, True)
        result_list.extend(corrected_ing_list)
        plurals_list = addPluralsAndSingulars(corrected_ing_list, new_vocab_list)
        for ing in plurals_list:
            if ing not in result_list:
                result_list.append(ing)
        processed_string = " ".join(result_list)
    return processed_string



# ================================================
df = load_data("../data/final/full_recipes.csv")

st.subheader("Let's see some recipes!")

# A. Load the model using joblib
model = joblib.load('model_final.pkl')
new_vocab_list = joblib.load('custom_vocab.pkl')
vect = joblib.load('vect_mod.pkl')

# B1 - Categories
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

# B2 - Use desired ingredients and undesired ingredients
yes_ing = st.text_input('Enter the ingredients to include below, separated by commas', 'Chicken, Potato')
no_ing = st.text_input('Enter the ingredients to exclude below, separated by commas', '')

include_ing_string = category_string + " " + processInputString(yes_ing, new_vocab_list)
exclude_ing_string = processInputString(no_ing, new_vocab_list)

# st.write(category_string)
# st.write(include_ing_string)
# st.write(exclude_ing_string)

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

if updated_ing_tx is not None:
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
            st.write(":blue[Ingredients]")
            for food in ingredients:
                st.write(food)
            st.write(":blue[Steps]")
            st.write(steps)



###################
# Test Cases
# Add only beetroot for include ing - No recipes shown only error message
# Add only category with no ingredients in inc and exc
# Exclude something with nothing in inclusion

# Test 2 word ingredients - Peanut butter
# Test spell checker - Peanut butter, brocoli
# Test pluralization - Chicken Potato
# Blue bleu

# Check heavy cream as an ingredient - error message heavies
# include ingredients empty is giving error.
###################