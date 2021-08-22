from src import clean_text, load_csv, finding_cosine_scores, top_five_questions_index

df = load_csv()
text = input("Enter your Python related question: ")

clean_sentence = clean_text(text)
all_similarity_scores = finding_cosine_scores(df, clean_sentence)
questions_index = top_five_questions_index(all_similarity_scores)

print("\nThe top 5 questions similar to the input question are :\n")
for count, index in enumerate(questions_index):
    print(count + 1, "\t", df['Title'][index])
