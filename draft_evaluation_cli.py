import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_replies(original_email, human_reply, gpt_reply):
    vectorizer = TfidfVectorizer()
    docs = [original_email, human_reply, gpt_reply]
    tfidf_matrix = vectorizer.fit_transform(docs)

    similarity_human = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    similarity_gpt = cosine_similarity(tfidf_matrix[0], tfidf_matrix[2])[0][0]

    print("\n=== Reply Evaluation Results ===")
    print(f",Human Reply Similarity: {similarity_human:.4f}")
    print(f"GPT Reply Similarity:   {similarity_gpt:.4f}")

    if similarity_gpt > similarity_human:
        print("GPT reply is more similar to the original.")
    elif similarity_gpt < similarity_human:
        print("Human reply is more similar to the original.")
    else:
        print("Both replies are equally similar.")

def interactive_mode():
    print("\n--- INTERACTIVE EMAIL REPLY EVALUATOR ---")
    original = input("Enter the ORIGINAL email:\n> ")
    human = input("\nEnter the HUMAN reply:\n> ")
    gpt = input("\nEnter the GPT-generated reply:\n> ")
    evaluate_replies(original, human, gpt)

def demo_mode():
    print("\n--- DEMO MODE ---")
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

def main():
    print("\n Welcome to the Email Draft Evaluator!")
    print("1. Run in DEMO mode")
    print("2. Run in INTERACTIVE mode")
    choice = input("Choose an option [1/2]: ") 

    if choice == "1":
        demo_mode()
    elif choice == "2":
        interactive_mode()
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
