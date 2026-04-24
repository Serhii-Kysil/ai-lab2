"""
Завдання 2.2.3. SVM із сигмоїдальним ядром
"""

import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

input_file = './income_data.txt'
X = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line[:-1].split(', ')
        if len(data) < 15:
            continue
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Сигмоїдальне ядро
print("=" * 50)
print("SVM із сигмоїдальним ядром")
print("=" * 50)
classifier_sig = SVC(kernel='sigmoid', random_state=0)
classifier_sig.fit(X_train, y_train)
y_pred_sig = classifier_sig.predict(X_test)

print(f"Accuracy:  {round(100 * accuracy_score(y_test, y_pred_sig), 2)}%")
print(f"Precision: {round(100 * precision_score(y_test, y_pred_sig, average='weighted'), 2)}%")
print(f"Recall:    {round(100 * recall_score(y_test, y_pred_sig, average='weighted'), 2)}%")
print(f"F1 Score:  {round(100 * f1_score(y_test, y_pred_sig, average='weighted'), 2)}%")
print("\nДокладний звіт:")
print(classification_report(y_test, y_pred_sig, target_names=['<=50K', '>50K']))