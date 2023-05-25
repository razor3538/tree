from random import randrange
from csv import reader
import graphviz
import matplotlib.pyplot as plt

dot = graphviz.Digraph(comment='The Round Table')


def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = list(lines)
    return dataset


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def cross_validation_split(dataset, n_folds):  # разделение набора данных для тестирования и обучения модели
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def accuracy_metric(actual, predicted):  # точность предсказаний в %
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def evaluate_algorithm(dataset, algorithm, n_folds,
                       *args):  # оцениваем точность алгоритма на наборе данных с использованием cross_validation_split
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)

        count = 0
        countMass = []

        for i in actual:
            count += 1
            countMass.append(count)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(countMass, predicted, color='tab:red', alpha=0.4, label='Точка отклонения')
        ax.scatter(countMass, actual, color='tab:blue', alpha=1, label='Настоящий результат')

        plt.title("Критерий отклонения предсказанных результатов от актуальных")
        plt.legend()

        plt.show()
        scores.append(accuracy)
    return scores


def test_split(index, value, dataset):  # дерево решений, разбиваем данные на 2 группы по заданному индексу и значению
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def gini_index(groups, classes):  # алгоритм критерия Джини
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0

    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)

    return gini


def get_split(dataset):  # выбираем наилучшее разбиение на основе критерия Джини (минимален)
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None

    giniMass = []
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini != 0.0:
                giniMass.append(gini)

            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups

    count = 0
    countMass = []

    for i in giniMass:
        count += 1
        countMass.append(count)

    if len(giniMass) != 0:
        plt.scatter(countMass, giniMass, color='tab:blue')

        plt.title("График Джинни")
        plt.show()

    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def to_terminal(group):  # определяем метку класса для листовой вершины
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def split(node, max_depth, min_size, depth):  # разбиваем узел на 2 поддерева
    left, right = node['groups']
    del (node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


def build_tree(train, max_depth, min_size):  # строим дерево решений на основе обучающего набора данных
    root = get_split(train)
    split(root, max_depth, min_size, 1)

    return root


def predict(node, row):  # предсказание метки класса на основе построенного дерева решений
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def decision_tree(train, test, max_depth, min_size):  # делаем предсказание по построенному дереву
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return (predictions)


filename = 'tree.csv'
dataset = load_csv(filename)

for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)

n_folds = 2
max_depth = 5  # максимальная глубина
min_size = 10  # минимальная размерность

scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)

print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

dot.render('doctest-output/round-table.gv').replace('\\', '/')
'doctest-output/round-table.gv.pdf'




