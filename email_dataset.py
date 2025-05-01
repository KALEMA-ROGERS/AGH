import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class EmailDatasetProcessor:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

    def clean_data(self):
        # Drop null rows in the 'text' column
        self.df.dropna(subset=['text'], inplace=True)

        # Add a binary label: if 'urgent' or 'action needed' in the text -> urgent
        self.df['label'] = self.df['text'].apply(
            lambda x: 'urgent' if any(word in x.lower() for word in [
                                      'urgent', 'action needed']) else 'normal'
        )

    def vectorize(self):
        tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        # Use 'text' column for vectorization
        X = tfidf.fit_transform(self.df['text'])
        y = self.df['label']  # Use 'label' column for target variable
        return X, y

    def load_dataset(self):
        self.clean_data()
        return self.vectorize()


if __name__ == "__main__":
    processor = EmailDatasetProcessor(
        "C:/Users/USER/Desktop/lablab_ai/AGH/processed_emails.csv")
    X, y = processor.load_dataset()
    print("Dataset ready:", X.shape, y.shape)
