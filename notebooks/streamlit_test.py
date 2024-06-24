### KICKOFF - CODING AN APP IN STREAMLIT

### import libraries
import pandas as pd
import streamlit as st
import joblib
import cust_tokenizer


# st.write('Streamlit is an open-source app framework for Machine Learning and Data Science teams. For the docs, please click [here](https://docs.streamlit.io/en/stable/api.html). \
#    This is is just a very small window into its capabilities.')


#######################################################################################################################################
### LAUNCHING THE APP ON THE LOCAL MACHINE
### 1. Save your *.py file (the file and the dataset should be in the same folder)
### 2. Open git bash (Windows) or Terminal (MAC) and navigate (cd) to the folder containing the *.py and *.csv files
### 3. Execute... streamlit run <name_of_file.py>
### 4. The app will launch in your browser. A 'Rerun' button will appear every time you SAVE an update in the *.py file



#######################################################################################################################################
### Create a title

st.title("SmartRecipes - A Recipe Recommender")

# Press R in the app to refresh after changing the code and saving here

### You can try each method by uncommenting each of the lines of code in this section in turn and rerunning the app

### You can also use markdown syntax.
#st.write('# Our last morning kick off :sob:')

### To position text and color, you can use html syntax
#st.markdown("<h1 style='text-align: center; color: blue;'>Our last morning kick off</h1>", unsafe_allow_html=True)


#######################################################################################################################################
### DATA LOADING

### A. define function to load data
# @st.cache_data # <- add decorators after tried running the load multiple times
def load_data(path):

    df = df = pd.read_csv(path)

    # Streamlit will only recognize 'latitude' or 'lat', 'longitude' or 'lon', as coordinates

    # df = df.rename(columns={'Start Station Latitude': 'lat', 'Start Station Longitude': 'lon'})
    # df['Start Time'] = pd.to_datetime(df['Start Time'])      # reset dtype for column

    return df

### B. Load first 50K rows
df = load_data("../data/interim/full_recipes_cleaned_2.csv")

### C. Display the dataframe in the app
st.dataframe(df.sample(5))


#######################################################################################################################################
### STATION MAP

# st.subheader('Location Map - NYC bike stations')

# st.map(df)


#######################################################################################################################################
### DATA ANALYSIS & VISUALIZATION

### B. Add filter on side bar after initial bar chart constructed

# st.sidebar.subheader("Usage filters")
# round_trip = st.sidebar.checkbox('Round trips only')

# if round_trip:
#     df = df[df['Start Station ID'] == df['End Station ID']]


# ### A. Add a bar chart of usage per hour

# st.subheader("Daily usage per hour")

# counts = df["Start Time"].dt.hour.value_counts()
# st.bar_chart(counts)





### The features we have used here are very basic. Most Python libraries can be imported as in Jupyter Notebook so the possibilities are vast.
#### Visualizations can be rendered using matplotlib, seaborn, plotly etc.
#### Models can be imported using *.pkl files (or similar) so predictions, classifications etc can be done within the app using previously optimized models
#### Automating processes and handling real-time data


#######################################################################################################################################
### MODEL INFERENCE

st.subheader("Using pretrained models with user input")

# A. Load the model using joblib
model = joblib.load('model_joblib.pkl')
ing_mat = joblib.load('ing_mat.pkl')
vect = joblib.load('vect.pkl')

# B. Set up input field and run model
# B1 - Recipe Name as input
# text = st.text_input('Enter the recipe name', 'Chicken Parmesan')
# recipe = df[df['title'] == text].index
# distances, indices = model.kneighbors(ing_mat[recipe])


# B2 - Ingredient list as input
# text = st.text_input('Enter the ingredients below', 'Chicken, Parmesan, Breadcrumbs')
# textSeries = pd.Series(text)
# textSeriesTransformed = vect.transform(textSeries)
# distances, indices = model.kneighbors(textSeriesTransformed)

# B3 - Use desired ingredients and undesired ingredients
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
    st.write(f"{name}  :  {distance}  :  {rating}")

# prediction = 1
# if prediction == 1:
#     st.write('We predict that this is a positive review!')
# else:
#     st.write('We predict that this is a negative review!')



#######################################################################################################################################
### Streamlit Advantages and Disadvantages

# st.subheader("Streamlit Advantages and Disadvantages")
# st.write('**Advantages**')
# st.write(' - Easy, Intuitive, Pythonic')
# st.write(' - Free!')
# st.write(' - Requires no knowledge of front end languages')
# st.write('**Disadvantages**')
# st.write(' - Apps all look the same')
# st.write(' - Not very customizable')
# st.write(' - A little slow. Not good for MLOps, therefore not scalable')
