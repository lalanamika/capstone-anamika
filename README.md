## SmartRecipes - a recipe recommender
## Capstone Project - BrainStation Data Science Bootcamp - Apr - Jul 2024
==========================================================================

### Table of Contents
- [Project Overview](#project-overview)
  * [The Problem Area](#the-problem-area)
  * [The User](#the-user)
  * [The Idea](#the-idea)
  * [The Impact](#the-impact)
  * [The Solution](#the-solution)
  * [The Data](#the-data)
- [Project Flowchart](#project-flowchart)
- [Project Organization](#project-organization)


### Project Overview

#### The Problem Area
Have you ever tried cooking a new recipe, and had leftover ingredients that you had no idea how to use, so you ended up throwing them away? This is a problem that I face often and I was curious to see whether I could build a recipe recommender using ML that could help with this.

#### The User
Home cooks and Food enthusiasts: Individuals who enjoy cooking at home and are looking for new recipes to try.

#### The Idea
Using machine learning, how might we “recommend food recipes” such that we can:

Reduce food waste (by providing ideas on how to use the ingredients they have or suggesting the quantities of ingredients that they will have to buy).

#### The Impact

Reduced Food Waste: The system can help reduce food waste by promoting efficient use of ingredients and leftovers. This contributes to sustainability efforts and addresses a pressing societal concern.

#### The Solution
SmartRecipes is a recipe recommendation system built with the goal of reducing food waste. 
    - The dataset has about 20,000 recipes. 
    - This is a content based recommendation system.
    - The recommender gives recipes from the dataset, ranked by closest match and less ingredients.

This is an unsupervised learning problem as we do not have a target variable. We will evaluate the quality of results by manually inspecting the ingredients in the results (or eventually writing a script for evaluation).

Since we are dealing with text data, we will use `CountVectorizer` to preprocess the data.

For modeling, we will be using scikit-learn's `NearestNeighbors` unsupervised learner to find the cosine similarity between the user input and recipes in the dataset.

We will also build a Streamlit app in which users can enter ingredients and the recommender returns the top 10 recipes with the relevant ingredients.

#### The Data
When looking for datasets, I was looking for the following information:
- Contains list of ingredients
- Contains quantity of ingredients
- Contains nutrition information / calorie information
- Contains ratings for the recipe
- Contains recipe steps
- Contains serving size
- Contains diet type (Vegan, Gluten-free etc.)
- Contains cuisine type (Italian, Indian etc.)

We will use the dataset from Kaggle - https://www.kaggle.com/datasets/hugodarwood/epirecipes.
This contains 'list of ingredients', 'measurements for ingredients', 'calories', 'ratings', 'steps for the recipe', 'some categories like vegetarian'.

**Data Dictionary**
| Column      | Non-Null Count | Dtype               | Description                                     |
|-------------|----------------|---------------------|-------------------------------------------------|
| ------      | -------------- | -----               | ------                                          |
| title       | 15969 non-null | object              | Title of the recipe                             |
| directions  | 15969 non-null | object              | Steps for the recipe                            |
| ingredients | 15969 non-null | object              | Ingredients plus description of how to cut them |
| categories  | 15969 non-null | object              | Array of categories                             |
| calories    | 15969 non-null | float64             | Calories                                        |
| rating      | 15969 non-null | float64             | Rating on a scale of 0 to 5                     |
| desc        | 10636 non-null | object              | Extra tidbits about the recipe etc.             |
| date        | 15969 non-null | datetime64[ns, UTC] | Date the recipe was created                     |
| sodium      | 15967 non-null | float64             | Sodium content                                  |
| fat         | 15901 non-null | float64             | Fat content                                     |
| protein     | 15922 non-null | float64             | Protein content                                 |


After removing null values and duplicate rows, there are 14526 recipes. This will be the size of the dataset we will do the modeling on.

### Project Flowchart
![High_Level_Flowchart](https://github.com/lalanamika/capstone-anamika/blob/main/visualizations/High_Level_Flowchart.png)


### Project Organization


* `data`
    - `raw` folder contains the relevant data copied from Kaggle.
    - `interim` folder contains the words excluded during custom tokenization as well as the cleaned datasets.
    - `final` folder contains the csv file after cleaning and feature engineering.

* `model`
    - joblib dump of final model, custom vocabulary, and vectorizer.

* `notebooks`
    - `1-EDA-Cleaning.ipynb` - Cleaning the dataset.
    - `2-Pre-processing.ipynb` - Using CountVectorizer to tokenize the data, and NearestNeighbors to do preliminary modelling.
    - `3-Initial-Modelling-Experiments.ipynb` - Experiments with different custom vocabulary, figuring out spell checker etc.
    - `4-Final-Modelling.ipynb` - Final modeling using a custom vocabulary.
    - `cust_tokenizer.py` - Module for custom tokenization. This is used by the notebooks and the Streamlit app.
 
* `visualizations`
    - Any flowcharts, or visualizations created during the project.

* `Streamlit`
    - `streamlit_app.py` - Streamlit app for the recipe recommender.
    - Also contains a screenshot of the Streamlit app that was built.

* `.gitignore`
    - Part of Git, includes files and folders to be ignored by Git version control

* `README.md`
    - Project landing page (this page)

* `LICENSE`
    - Project license
