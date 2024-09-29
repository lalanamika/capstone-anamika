from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

# Remove units of measurements such as teaspoons, cups, ounces etc. Full list at https://en.wikibooks.org/wiki/Cookbook:Units_of_measurement
# measurements = set(line.strip() for line in open('../data/interim/measurement_list.txt'))
measurements = set(line.strip() for line in open('C:/Users/anami/Downloads/Data_Science/capstone-anamika/data/interim/measurement_list.txt'))

# Remove extra adjectives like 'baked', 'thawed', 'cleaned' etc.
extra_adjectives = set(line.strip() for line in open('C:/Users/anami/Downloads/Data_Science/capstone-anamika/data/interim/extra_adjectives_list.txt'))

# Remove some extra words like 'assorted', 'approximately' etc. QUESTION: Is there a smart way to remove the top 100 such words?
extra_words = set(line.strip() for line in open('C:/Users/anami/Downloads/Data_Science/capstone-anamika/data/interim/extra_words_list.txt'))


"""
custom tokenizer examples are in 0604_nlp_part2a_beta and 0531_text_vect_redux and 0531_Text_Data
"""
my_stops = set(ENGLISH_STOP_WORDS) | measurements | extra_adjectives | extra_words

def my_tokenizer(text):
    # convert to lowercase
    text = text.lower()
    # break into characters and weed out punctuation etc.  (include space!)
    chars = list(char for char in text if char in "abcdefghijklmnopqrstuvwxyz ")
    # make back into a single string
    text = "".join(chars)
    # break into words and weed out stop words and short words < 3 characters
    text = list(word for word in text.split() if word not in my_stops and len(word) >=3)
    return text

if __name__ == "__main__":
    result = my_tokenizer("Called from main")
    print(result)