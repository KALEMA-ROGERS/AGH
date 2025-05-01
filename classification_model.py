from email_dataset import EmailDatasetProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report


def main():
    # Load and process dataset
    processor = EmailDatasetProcessor(
        "C:/Users/USER/Desktop/lablab_ai/AGH/processed_emails.csv")
    X, y = processor.load_dataset()

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train model
    model = LogisticRegression(
        max_iter=1000,
        solver='liblinear',
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)

    print("\n=== Model Evaluation Metrics ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(
        f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
