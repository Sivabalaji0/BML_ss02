"""
Handwritten digit recognition - training, model comparison, hyperparameter tuning.
Data set : scikit-learn 'digits' (1797 images, 8x8 pixels, classes 0-9). Bundled with
           scikit-learn, no download needed. For full MNIST (28x28) swap load_data().
Outputs  : model/model.joblib, model/metrics.json, model/weights.json, app.html
"""
import json, joblib, numpy as np
from digit_preprocess import preprocess_np, render
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)

RS = 42

def load_data():
    d = load_digits()
    return d.data / 16.0, d.target          # pixels scaled to 0..1

X, y = load_data()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
cv = StratifiedKFold(5, shuffle=True, random_state=RS)

# ---------------------------------------------------------------- model comparison
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "KNN":                 KNeighborsClassifier(),
    "SVM (RBF)":           SVC(random_state=RS),
    "Random Forest":       RandomForestClassifier(random_state=RS),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64,), max_iter=800, random_state=RS),
}
comparison = {}
for name, m in models.items():
    cv_acc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="accuracy").mean()
    m.fit(X_tr, y_tr); p = m.predict(X_te)
    comparison[name] = {"cv_accuracy": round(cv_acc, 4),
                        "test_accuracy": round(accuracy_score(y_te, p), 4),
                        "precision_macro": round(precision_score(y_te, p, average="macro"), 4),
                        "recall_macro": round(recall_score(y_te, p, average="macro"), 4),
                        "f1_macro": round(f1_score(y_te, p, average="macro"), 4)}
    print(f"{name:22s}", comparison[name])

# ---------------------------------------------------------------- hyperparameter tuning
# Application model: neural network (runs in the browser app), tuned by grid search.
grid = GridSearchCV(
    MLPClassifier(max_iter=800, random_state=RS),
    {"hidden_layer_sizes": [(32,), (64,), (128,)], "alpha": [1e-4, 1e-3, 1e-2],
     "learning_rate_init": [1e-3, 5e-3]},
    cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X_tr, y_tr)
best = grid.best_estimator_
print("\nBest MLP params:", grid.best_params_, "| CV acc:", round(grid.best_score_, 4))

# ---------------------------------------------------------------- deployment-aligned training
# The app crops/centres a drawn digit before prediction. Train on digits processed the same
# way (training split only, so no test leakage) so the browser input matches the training data.
rng = np.random.default_rng(RS)
def deploy_view(X, copies):
    out, lab = [], []
    for img in X:
        for _ in range(copies):
            k = int(rng.integers(20, 35)); s = min(8 * k, 280)
            f = preprocess_np(render(img, k, int(rng.integers(0, 280 - s + 1)), int(rng.integers(0, 280 - s + 1))))
            out.append(f)
    return np.array(out)
X_aug = np.vstack([X_tr, deploy_view(X_tr, 3)]); y_aug = np.concatenate([y_tr, np.repeat(y_tr, 3)])
best = MLPClassifier(max_iter=800, random_state=RS, **grid.best_params_).fit(X_aug, y_aug)
X_te_dep = deploy_view(X_te, 1)                      # test images pushed through the app pipeline

# ---------------------------------------------------------------- final evaluation
p = best.predict(X_te); p_dep = best.predict(X_te_dep)
metrics = {
    "dataset": "sklearn digits (1797 x 8x8, 10 classes)",
    "best_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in grid.best_params_.items()},
    "cv_accuracy": round(grid.best_score_, 4),
    "test_accuracy": round(accuracy_score(y_te, p), 4),
    "test_accuracy_app_pipeline": round(accuracy_score(y_te, p_dep), 4),
    "precision_macro": round(precision_score(y_te, p, average="macro"), 4),
    "recall_macro": round(recall_score(y_te, p, average="macro"), 4),
    "f1_macro": round(f1_score(y_te, p, average="macro"), 4),
    "confusion_matrix": confusion_matrix(y_te, p).tolist(),
    "model_comparison": comparison,
}
print("\nFinal MLP - test accuracy (original 8x8):", metrics["test_accuracy"],
      "| through the app's draw->preprocess pipeline:", metrics["test_accuracy_app_pipeline"])
print(classification_report(y_te, p, digits=3))
joblib.dump(best, "model/model.joblib")
json.dump(metrics, open("model/metrics.json", "w"), indent=2)

# export weights for the browser app
w = {"W0": np.round(best.coefs_[0], 5).tolist(), "b0": np.round(best.intercepts_[0], 5).tolist(),
     "W1": np.round(best.coefs_[1], 5).tolist(), "b1": np.round(best.intercepts_[1], 5).tolist()}
json.dump(w, open("model/weights.json", "w"))
tpl = open("app_template.html").read()
open("app.html", "w").write(tpl.replace("__WEIGHTS__", json.dumps(w)))
np.save("model/test_X.npy", X_te); np.save("model/test_y.npy", y_te)
np.save("model/test_X_app.npy", X_te_dep)
