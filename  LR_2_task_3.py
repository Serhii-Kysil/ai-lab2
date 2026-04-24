"""
Завдання 2.3. Порівняння якості класифікаторів — набір даних Iris
"""

import numpy as np
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_iris

# ======== КРОК 1: ЗАВАНТАЖЕННЯ ТА ВИВЧЕННЯ ДАНИХ ========
iris_dataset = load_iris()
print("Ключі iris_dataset:\n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Назви відповідей: {}".format(iris_dataset['target_names']))
print("Назва ознак:\n{}".format(iris_dataset['feature_names']))
print("Тип масиву data: {}".format(type(iris_dataset['data'])))
print("Форма масиву data: {}".format(iris_dataset['data'].shape))
print("Перші 5 прикладів:\n{}".format(iris_dataset['data'][:5]))
print("Тип масиву target: {}".format(type(iris_dataset['target'])))
print("Відповіді:\n{}".format(iris_dataset['target']))

# Завантаження через pandas з URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

print("\nРозмір датасету:", dataset.shape)
print("\nПерші 20 рядків:")
print(dataset.head(20))
print("\nСтатистичне зведення:")
print(dataset.describe())
print("\nРозподіл за класами:")
print(dataset.groupby('class').size())

# ======== КРОК 2: ВІЗУАЛІЗАЦІЯ ========
# Діаграма розмаху
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.suptitle('Діаграма розмаху атрибутів')
pyplot.tight_layout()
pyplot.savefig('boxplot_iris.png', dpi=100)
pyplot.show()

# Гістограма
dataset.hist()
pyplot.suptitle('Гістограма розподілу атрибутів')
pyplot.tight_layout()
pyplot.savefig('histogram_iris.png', dpi=100)
pyplot.show()

# Матриця діаграм розсіювання
scatter_matrix(dataset)
pyplot.suptitle('Матриця діаграм розсіювання')
pyplot.tight_layout()
pyplot.savefig('scatter_matrix_iris.png', dpi=100)
pyplot.show()

# ======== КРОК 3: НАВЧАЛЬНА ТА ТЕСТОВА ВИБІРКИ ========
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]

X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1
)

# ======== КРОК 4: ПОРІВНЯННЯ АЛГОРИТМІВ ========
models = [
    ('LR',   LogisticRegression(solver='liblinear', multi_class='ovr')),
    ('LDA',  LinearDiscriminantAnalysis()),
    ('KNN',  KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB',   GaussianNB()),
    ('SVM',  SVC(gamma='auto')),
]

results = []
names = []
print("\nРезультати крос-валідації (accuracy):")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean():.4f} (±{cv_results.std():.4f})')

# Порівняльний графік
pyplot.boxplot(results, labels=names)
pyplot.title('Порівняння алгоритмів класифікації (Iris)')
pyplot.ylabel('Accuracy')
pyplot.savefig('algorithm_comparison_iris.png', dpi=100)
pyplot.show()

# ======== КРОКИ 6-7: ПРОГНОЗ І ОЦІНКА ========
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print("\nОцінка якості SVM на контрольній вибірці:")
print(f"Accuracy: {accuracy_score(Y_validation, predictions):.4f}")
print("\nМатриця помилок:")
print(confusion_matrix(Y_validation, predictions))
print("\nЗвіт про класифікацію:")
print(classification_report(Y_validation, predictions))

# ======== КРОК 8: ПЕРЕДБАЧЕННЯ ДЛЯ НОВОЇ КВІТКИ ========
X_new = np.array([[5, 2.9, 1, 0.2]])
print("Форма масиву X_new: {}".format(X_new.shape))

prediction = model.predict(X_new)
print(f"Прогноз (числовий код): {prediction}")
# Відображення назви класу через iris_dataset
label_map = {name: i for i, name in enumerate(iris_dataset['target_names'])}
reverse_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
class_names = list(set(y))
print(f"Спрогнозована мітка: {prediction[0]}")
print(f"Висновок: квітка належить до сорту '{prediction[0]}'")