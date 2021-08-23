from src import clean_text, load_csv, finding_cosine_scores, top_five_questions_index

# Load the csv file in the dataframe
df = load_csv()

# Take the user input
text = input("Enter your Python related question: ")

# Preprocess the text, and calculate similarity score between the texts
clean_sentence = clean_text(text)
all_similarity_scores = finding_cosine_scores(df, clean_sentence)

# Find the top five question index and output top 5 similar questions
questions_index = top_five_questions_index(all_similarity_scores)
print("\nThe top 5 questions similar to the input question are :\n")
for count, index in enumerate(questions_index):
    print(count + 1, "\t", df['Title'][index])
