import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import random

from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


SEED = 42  # global random seed
np.random.seed(SEED)
random.seed(SEED)


features = np.load("features.npy", allow_pickle=True)  # load feature vectors
labels = np.load("labels.npy", allow_pickle=True)      # load class labels

assert len(features) == len(labels), "Mismatch between features and labels"

print(f"Total Samples: {len(features)}")
print(f"Unique Classes (Raw): {len(np.unique(labels))}")



# organize samples perr class
class_samples = defaultdict(list)  # group features per class

for x, y in zip(features, labels):
    class_samples[y].append(x)

X_train, y_train = [], []
X_test, y_test = [], []
skipped = 0

for label in sorted(class_samples.keys()):  # deterministic ordering
    samples = class_samples[label]

    if len(samples) < 2:
        skipped += 1
        continue  # skip classes with <2 samples

    X_test.append(samples[0])               # first sample -> test
    y_test.append(label)

    X_train.extend(samples[1:])             # remaining -> train
    y_train.extend([label] * (len(samples) - 1))

X_train = np.array(X_train)
X_test = np.array(X_test)

print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")
print(f"Classes Used: {len(set(y_train))}")
print(f"Skipped Classes: {skipped}")

assert X_train.shape[1] == X_test.shape[1], "Feature dimension mismatch"

#encode labels
le = LabelEncoder()                         # initialize encoder
y_train_enc = le.fit_transform(y_train)    # encode training labels
y_test_enc = le.transform(y_test)          # encode testing labels


# train Model
clf = RandomForestClassifier(
    n_estimators=300,        # number of trees
    max_depth=None,          # allow full tree growth
    min_samples_split=2,     # minimum split size
    min_samples_leaf=1,      # minimum leaf size
    bootstrap=True,          # use bootstrap samples
    random_state=SEED,
    n_jobs=-1                # use all CPU cores
)

clf.fit(X_train, y_train_enc)  # train classifier


# evaluation
y_pred = clf.predict(X_test)  # predict test set

acc = accuracy_score(y_test_enc, y_pred)  # compute top-1 accuracy
print(f"\nTop-1 Accuracy: {acc:.4f}")

print("\nClassification Report:\n")
print(classification_report(
    y_test_enc,
    y_pred,
    target_names=le.classes_,
    digits=4,
    zero_division=0
))


# Confusion Matrix (Normalized)
cm = confusion_matrix(y_test_enc, y_pred, normalize="true")  # normalize per class

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    cmap="coolwarm",
    annot=False
)
plt.title("Normalized Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

joblib.dump(clf, "speaker_recognition_model.pkl")  # save trained model
joblib.dump(le, "label_encoder.pkl")               # save label encoder

print("\nModel and LabelEncoder saved successfully.")