"""
Завдання 2.1. Класифікація за допомогою машин опорних векторів (SVM)
Прогнозування доходу фізичної особи (>50K або <=50K)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

# Вхідний файл, який містить дані
input_file = './income_data.txt'

# Читання даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue

        # Розбиття рядка за комою з пробілом
        data = line[:-1].split(', ')

        if len(data) < 15:
            continue

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

print(f"Клас <=50K: {count_class1} точок")
print(f"Клас >50K:  {count_class2} точок")

# Перетворення на масив numpy
X = np.array(X)

# Перетворення рядкових даних на числові (кодування міток)
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

# Розбиття на навчальну та тестову вибірки (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5
)

# Створення та навчання SVM-класифікатора
classifier = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=2000))
classifier.fit(X_train, y_train)

# Прогноз на тестовій вибірці
y_test_pred = classifier.predict(X_test)

# Обчислення показників якості
f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print(f"\nF1 score (крос-валідація): {round(100 * f1.mean(), 2)}%")

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1_test = f1_score(y_test, y_test_pred, average='weighted')

print(f"Accuracy  (Акуратність): {round(100 * accuracy, 2)}%")
print(f"Precision (Точність):    {round(100 * precision, 2)}%")
print(f"Recall    (Повнота):     {round(100 * recall, 2)}%")
print(f"F1 Score:                {round(100 * f1_test, 2)}%")
print("\nДокладний звіт:")
print(classification_report(y_test, y_test_pred, target_names=['<=50K', '>50K']))

# Передбачення результату для тестової точки даних
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
              '0', '0', '40', 'United-States']

# Кодування тестової точки даних
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]])[0])
        count += 1

input_data_encoded = np.array(input_data_encoded)

# Прогноз для тестової точки
predicted_class = classifier.predict(input_data_encoded.reshape(1, -1))
print(f"\nТестова точка належить до класу: {label_encoder[-1].inverse_transform(predicted_class)[0]}")