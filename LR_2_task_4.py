"""
Завдання 2.4. Порівняння якості класифікаторів для набору income_data.txt
"""

import numpy as np
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Завантаження та підготовка даних
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

X_data = X_encoded[:, :-1].astype(int)
y_data = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=5
)

# Порівняння алгоритмів
models = [
    ('LR',   LogisticRegression(solver='liblinear', max_iter=1000)),
    ('LDA',  LinearDiscriminantAnalysis()),
    ('KNN',  KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB',   GaussianNB()),
    ('SVM',  SVC(kernel='rbf', gamma='auto')),
]

results = []
names = []
print("Результати крос-валідації (accuracy) для income_data:")
for name, model in models:
    kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean():.4f} (±{cv_results.std():.4f})')

# Порівняльний графік
pyplot.boxplot(results, labels=names)
pyplot.title('Порівняння алгоритмів класифікації (Income Data)')
pyplot.ylabel('Accuracy')
pyplot.savefig('algorithm_comparison_income.png', dpi=100)
pyplot.show()

# Детальна оцінка найкращої моделі (зазвичай LDA або KNN для цього датасету)
print("\n--- Детальна оцінка кожного класифікатора на тестовій вибірці ---")
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} — Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))