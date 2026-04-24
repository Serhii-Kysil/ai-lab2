"""
Завдання 2.5. Класифікація даних лінійним класифікатором Ridge
Виправлений код з усіма помилками
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split  # ВИПРАВЛЕНО: додано імпорт
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Завантаження даних
iris = load_iris()
X, y = iris.data, iris.target

# ВИПРАВЛЕНО: використано X_train, X_test (не Xtrain/Xtest) консистентно
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Класифікатор Ridge
# tol=1e-2 — допуск для критерію зупинки (точність збіжності)
# solver="sag" — метод стохастичного усередненого градієнту (ефективний для великих даних)
clf = RidgeClassifier(tol=1e-2, solver="sag")
clf.fit(X_train, y_train)

# ВИПРАВЛЕНО: X_test (а не Xtest)
y_pred = clf.predict(X_test)

# Показники якості
print("=" * 50)
print("Показники якості класифікатора Ridge")
print("=" * 50)
print('Accuracy (Акуратність):',
      np.round(metrics.accuracy_score(y_test, y_pred), 4))
print('Precision (Точність):',
      np.round(metrics.precision_score(y_test, y_pred, average='weighted'), 4))
print('Recall (Повнота):',
      np.round(metrics.recall_score(y_test, y_pred, average='weighted'), 4))
print('F1 Score:',
      np.round(metrics.f1_score(y_test, y_pred, average='weighted'), 4))
print('Cohen Kappa Score:',
      np.round(metrics.cohen_kappa_score(y_test, y_pred), 4))
print('Matthews Corrcoef:',
      np.round(metrics.matthews_corrcoef(y_test, y_pred), 4))

# ВИПРАВЛЕНО: порядок аргументів — спочатку y_test, потім y_pred
print('\n\t\tClassification Report:\n',
      metrics.classification_report(y_test, y_pred))

# Матриця помилок (теплова карта)
mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Справжній клас (true label)')
plt.ylabel('Передбачений клас (predicted label)')
plt.title('Матриця помилок — Ridge Classifier')
plt.tight_layout()
plt.savefig("Confusion.jpg")

# Збереження у SVG
f = BytesIO()
plt.savefig(f, format="svg")
print("\nМатрицю помилок збережено у файл Confusion.jpg")
plt.show()