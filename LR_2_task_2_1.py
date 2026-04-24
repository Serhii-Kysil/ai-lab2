"""
Завдання 2.2.1. SVM з поліноміальним ядром
ВИПРАВЛЕННЯ: зменшено max_datapoints до 5000 через квадратичну складність
SVC з нелінійними ядрами — O(n²..n³), на 50000 точках зависає назавжди
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5
)

print("=" * 50)
print("SVM з поліноміальним ядром (degree=3)")
print("=" * 50)

classifier_poly = SVC(kernel='poly', degree=3, random_state=0)
classifier_poly.fit(X_train, y_train)
y_pred_poly = classifier_poly.predict(X_test)

print(f"Accuracy:  {round(100 * accuracy_score(y_test, y_pred_poly), 2)}%")
print(f"Precision: {round(100 * precision_score(y_test, y_pred_poly, average='weighted'), 2)}%")
print(f"Recall:    {round(100 * recall_score(y_test, y_pred_poly, average='weighted'), 2)}%")
print(f"F1 Score:  {round(100 * f1_score(y_test, y_pred_poly, average='weighted'), 2)}%")
print("\nДокладний звіт:")
print(classification_report(y_test, y_pred_poly, target_names=['<=50K', '>50K']))