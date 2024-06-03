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
Contains list of ingredients
Contains quantity of ingredients
Contains nutrition information / calorie information
Contains ratings for the recipe
Contains recipe steps
Contains serving size
Contains diet type (Vegan, Gluten-free etc.)
Contains cuisine type (Italian, Indian etc.)

I checked various datasets for these things and tracked them in my [Datasets shortlist](https://docs.google.com/spreadsheets/d/1ldHpPRw_h2igZUgrDVV3-4N8vYuZXxyOC60sUJlbb7E/edit#gid=0)

Currently the following 2 datasets seem to be good candidates for this project.
1. https://www.kaggle.com/datasets/hugodarwood/epirecipes

**Pros:**
Contains ingredient measurements in the recipe.

**Cons:**
Small size ~ 16000 rows.
Kaggle License says Unknown.
Does not contain diet type (Vegan etc.) and cuisine type, but I have not been able to find any suitable dataset that contains that.
Does not contain serving size.


2. https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews

**Pros:**
Contains serving size.

**Cons:**
Does not have measurements for the ingredients.

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
