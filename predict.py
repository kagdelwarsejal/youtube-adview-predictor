import joblib
import numpy as np

# Load the trained model
model = joblib.load("artifacts/model.joblib")

# Feature order must match training
FEATURE_NAMES = [
    'views', 'likes', 'dislikes', 'comment_count', 
    'published_month', 'published_year', 
    'category_id', 'duration_seconds'
]

def predict(data):
    """
    data: dict containing feature values
    Example:
        {
            "views": 50000,
            "likes": 2000,
            "dislikes": 100,
            "comment_count": 300,
            "published_month": 5,
            "published_year": 2021,
            "category_id": 24,
            "duration_seconds": 210
        }
    """
    try:
        # Arrange features in correct order
        X = np.array([[data[feature] for feature in FEATURE_NAMES]])
        prediction = model.predict(X)
        return float(prediction[0])
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Example test run
    sample = {
        "views": 50000,
        "likes": 2000,
        "dislikes": 100,
        "comment_count": 300,
        "published_month": 5,
        "published_year": 2021,
        "category_id": 24,
        "duration_seconds": 210
    }
    print(predict(sample))
