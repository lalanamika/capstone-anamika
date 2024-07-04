### import libraries
import pandas as pd
import streamlit as st
import joblib
import cust_tokenizer

st.set_page_config("SmartRecipes", ":green_salad:", layout="wide")

# Custom HTML/CSS for the banner - for this the image has to be on a URL, not on local machine
custom_html = """
<div class="banner">
    <img src="https://cdn.stocksnap.io/img-thumbs/960w/peppers-vegetables_CSIVDF12OA.jpg" alt="Banner Image">
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
</style>
"""
# Display the custom HTML
st.components.v1.html(custom_html)

### To position text and color, you can use html syntax
st.markdown("<h1 style='text-align: center; color: blue;'>SmartRecipes</h1>", unsafe_allow_html=True)

# Load the dataset
@st.cache_data # <- add decorators after tried running the load multiple times
def load_data(path):

    df = df = pd.read_csv(path)
    return df

df = load_data("../data/final/full_recipes.csv")

st.subheader("Let's see some recipes!")

include_ing_list = []
exclude_ing_list = []

genre = st.radio(
    "Are you looking for any specific category of recipe?",
    ["No", ":green[Vegetarian] :leafy_green:", "Gluten-Free", "Dairy-Free", "Dessert :cake:"])

if genre == ":green[Vegetarian] :leafy_green:":
    include_ing_list.append("Vegetarian")
elif genre == "Gluten-Free":
    include_ing_list.append("Gluten free")
elif genre == "Dairy-Free":
    include_ing_list.append("Dairy Free")
elif genre == "Dessert :cake:":
    include_ing_list.append("Dessert")


# A. Load the model using joblib
model = joblib.load('model_final.pkl')
new_vocab_list = joblib.load('custom_vocab.pkl')
vect = joblib.load('vect_mod.pkl')

# B - Use desired ingredients and undesired ingredients
yes_ing = st.text_input('Enter the ingredients to include below', 'Chicken, Parmesan, Breadcrumbs')
no_ing = st.text_input('Enter the ingredients to exclude below', 'None')

yes_ing_series = pd.Series(yes_ing)
yes_ing_tx = vect.transform(yes_ing_series)

no_ing_series = pd.Series(no_ing)
no_ing_tx = (vect.transform(no_ing_series)) * -1

updated_ing_tx = yes_ing_tx + no_ing_tx

distances, indices = model.kneighbors(updated_ing_tx)

# C.  Print result
for i in range(0, 11):  # TODO: 11 should be made configurable and match the n-neighbors number
    name = df.loc[indices[0][i], ['title']].values[0]
    distance = (distances[0][i]).round(3)
    rating = df.loc[indices[0][i], ['rating']].values[0]
    ingredients = df.loc[indices[0][i], ['ingredientsStr']].values[0].split("',")
    steps = df.loc[indices[0][i], ['directionsStr']].values[0]
    st.write(f"{name}  :  {distance}  :  {rating}")
    with st.expander(name):
        st.write(ingredients)
        st.write(steps)
