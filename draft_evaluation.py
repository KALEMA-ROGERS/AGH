import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_replies(original_email, human_reply, gpt_reply):
    """
    Compares human and GPT replies to the original email using cosine similarity.
    """
    vectorizer = TfidfVectorizer()

    docs = [original_email, human_reply, gpt_reply]
    tfidf_matrix = vectorizer.fit_transform(docs)

    # Similarity scores
    similarity_human = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    similarity_gpt = cosine_similarity(tfidf_matrix[0], tfidf_matrix[2])[0][0]

    print("\n=== Reply Evaluation Results ===")
    print(f"Human Reply Similarity to Original: {similarity_human:.4f}")
    print(f"GPT Reply Similarity to Original:   {similarity_gpt:.4f}")

    if similarity_gpt > similarity_human:
        print("GPT reply is more similar to the original email.")
    elif similarity_gpt < similarity_human:
        print("Human reply is more similar to the original email.")
    else:
        print("Both replies are equally similar.")

def main():
    # input can be dynamically loaded from your dataset
    original_email = """
    Hello, I would like to schedule a meeting regarding the quarterly budget.
    Please let me know your availability this week.
    """

    human_reply = """
    Sure, I'm available on Tuesday or Thursday afternoon. Let me know what works best.
    """

    gpt_reply = """
    I am available this week for a meeting to discuss the quarterly budget. Please suggest a time.
    """

    evaluate_replies(original_email, human_reply, gpt_reply)

if __name__ == "__main__":
    main()
