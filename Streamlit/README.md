# Caveats of the Streamlit deployment

The app is deployed at [smartrecipes2.streamlit.app](https://smartrecipes2.streamlit.app/)

Source file for the Streamlit deployment: streamlit_app_2.py

Reason: streamlit_app.py has the original implementation of the site that runs properly on local machines, and uses the `pattern` package to handle singular and plurals of the ingredients. However this package was having a LOT of dependency problems when I was trying to Deploy the app on the Streamlit server. So I created streamlit_app_2.py and removed all references and usages of the `pattern` package. So the deployed app does not automatically consider the plural / singular version of the ingredients entered.

It took me a while to figure out how to make the Streamlit app reference files in other directories on the deployment server, so I copied the following files into this directory.
- `cust_tokenizer.py`
- `measurement_list.txt`
- `extra_adjectives_list.txt`
- `extra_words_list.txt`

I finally figured out that the way to reference other directories on the deployment server can be done by following the steps shown in the following example:
```
import os.path
path = os.path.dirname(__file__)
df = load_data(path+"/../data/final/full_recipes.csv")
```
