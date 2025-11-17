import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

X = np.load("C:\\Users\\Jonathan Philips\\Coding\\AI-Projects\\asl-detect\\asl_X.npy")
y = np.load("C:\\Users\\Jonathan Philips\\Coding\\AI-Projects\\asl-detect\\asl_y.npy")
class_names = np.load(
    "C:\\Users\\Jonathan Philips\\Coding\\AI-Projects\\asl-detect\\asl_classes.npy"
)

print("X:", X.shape)
print("y:", y.shape)
print("Classes:", class_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
print(f"\nTest accuracy: {acc:.4f}\n")

y_pred = clf.predict(X_test)
labels = list(range(len(class_names)))

print(classification_report(y_test, y_pred, labels=labels, target_names=class_names))

joblib.dump(
    {"model": clf, "class_names": class_names},
    "C:\\Users\\Jonathan Philips\\Coding\\AI-Projects\\asl-detect\\asl_landmark_model.pkl",
)
print("Saved model to asl_landmark_model.pkl")
