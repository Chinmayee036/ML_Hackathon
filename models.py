from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from .config import RANDOM_STATE


def build_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=RANDOM_STATE)
        ),
        "Gaussian Naive Bayes": make_pipeline(
            StandardScaler(),
            GaussianNB()
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Histogram Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.1, random_state=RANDOM_STATE
        ),
    }
