## Anamika Lal's Capstone Project
## BrainStation Data Science Bootcamp - Apr - Jul 2024
======================================================

### Project Overview

#### The Problem Area
I am interested in cooking and nutrition and want to build a recipe recommendation system such that it can help reduce food waste, promote mindful / conscious eating, and encourage users to try new recipes.

#### The User
Home cooks and Food enthusiasts: Individuals who enjoy cooking at home and are looking for new recipes to try.

Health-conscious consumers: Users who prioritize health and nutrition in their dietary choices.

Cultural explorers: Users interested in exploring diverse cuisines.

#### The Idea
At a high level, my idea is as follows:

Using machine learning, how might we “recommend food recipes” such that we can:

Reduce food waste (by providing ideas on how to use the ingredients they have or suggesting the quantities of ingredients that they will have to buy).

Promote mindful / conscious eating (by providing calorie info / nutrition info).

Encourage users to try new recipes (by recommending options based on their preferences). Some options could be:
Ingredients
Type of cuisine
Calorie count / Nutrition information
Vegan / Vegetarian / Gluten free etc.

##### Some thoughts about implementation.

What is the definition of a 'good'? Is the recipe highly rated by multiple users?

What would be a good recipe in terms of calories? Uses ingredients that the user is looking for and order by rating?

What would be a good recipe in terms of reducing food waste?
Could be a recipe that uses minimal ingredients (including the ingredient I am looking for).
Or if it calls for a can something, make sure it uses the full can.
If a recipe calls for a lot of fresh vegetables, but the recipe calls only for less portions, then it is not good in terms of food waste.


#### The Impact

Reduced Food Waste: The system can help reduce food waste by promoting efficient use of ingredients and leftovers. This contributes to sustainability efforts and addresses a pressing societal concern.

Promotion of Healthy Eating: By offering personalized recipe suggestions based on users' preferences and dietary restrictions, the recommendation system can encourage users to explore and adopt healthier eating habits. This aligns with the growing societal emphasis on wellness and nutrition.

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

A preliminary EDA showed that 54% of the recipes in the dataset have ratings > 4.0 (on a scale of 0 to 5).

This is an unsupervised learning problem as we do not have a target variable. We will evaluate the quality of results by manually inspecting the ingredients in the results (or eventually writing a script for evaluation).

Since we are dealing with text data, we will use `CountVectorizer` to preprocess the data.

For modeling, we will be using `NearestNeighbors` to find the cosine similarity between the user input and recipes in the dataset.

### Walkthrough Demo

...
...
...

### Project Flowchart

...
...
...

### Project Organization

...
...
...

* `data`
    - contains link to copy of the dataset (stored in a publicly accessible Google Drive folder)
    - saved copy of aggregated / processed data as long as those are not too large (> 10 MB)

* `model`
    - joblib dump of final model / model object

* `notebooks`
    - contains all final notebooks involved in the project

* `reports`
    - contains final report which summarises the project

* `references`
    - contains papers / tutorials used in the project

* `src`
    - Contains the project source code (refactored from the notebooks)

* `.gitignore`
    - Part of Git, includes files and folders to be ignored by Git version control

* `capstine_env.yml`
    - Conda environment specification

* `Makefile`
    - Automation script for the project

* `README.md`
    - Project landing page (this page)

* `LICENSE`
    - Project license

### Dataset

...
...
...

### Credits & References

...
...
...

--------
