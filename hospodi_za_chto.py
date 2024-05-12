import math
import random
import pandas as pd
from matplotlib import pyplot as plt
from copy import deepcopy
from sklearn.metrics import auc

print('1. Відкрити та зчитати файл з даними.')
data = pd.read_csv('KM-12-2.csv')

print('2. Визначити збалансованість набору даних. Вивести кількість об’єктів кожного класу.')
class_counts = data['GT'].value_counts()
print(class_counts) # Кількість об'єктів кожного класу однакова, ми можемо вважати набір даних збалансованим.


print('3а. Обчислити всі метрики (Accuracy, Precision, Recall, F-Scores, Matthews Correlation Coefficient,', 
      'Balanced Accuracy, Youden’s J statistics,  Area  Under  Curve  for  Precision-Recall  Curve,', 
      'Area  Under  Curve  for  Receiver  Operation  Curve)  для  кожної  моделі  при  різних значеннях порогу класифікатора', 
      '(крок зміни порогу 0.2).')

def show_graph_all_metrics_with_thresholds(metrics, model_name, thresholds):
    new_metrics = deepcopy(metrics)
    new_metrics.pop('FPR')
    new_metrics.pop('class_0')
    new_metrics.pop('class_1')

    plt.figure(figsize=(10, 6))
    for metric_name, values in new_metrics.items():
        plt.plot(thresholds, values, label=metric_name)

        max_value = max(values)
        max_thresholds = thresholds[values.index(max_value)]
        plt.scatter(max_thresholds, max_value, marker='X', color='red')
        plt.text(max_thresholds, max_value, f"({max_thresholds}, {round(max_value, 3)})", fontsize=10)

    plt.xlim([-0.01, 1.25])
    plt.ylim([-0.01, 1.07])

    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title(f'Metrics for {model_name[0][:7]}')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def calculate_metrics_data_frame(df, thresholds_step):
    list_of_thresholds = [round(i * thresholds_step, 4) for i in range(int(1 / thresholds_step) + 1)]

    for model in [('Model_1_0', 'Model_1_1'), ('Model_2_0', 'Model_2_1')]:
        all_metrics = {'accuracy': [], 
                       'precision': [], 
                       'recall': [], 
                       'f_scores': [], 
                       'MCC': [], 
                       'BA': [], 
                       'Y_J_statistics': [], 
                       'FPR': [], 
                       'class_0': [], 
                       'class_1': []}
        
        print("\n", model[:7])
        for threshold in list_of_thresholds:
            calculate_all_metrics(all_metrics, df, model, threshold)

        print('Thresholds: ', list_of_thresholds)
        for key, value in all_metrics.items():
            print(f'{key}: {value}')

        show_graph_all_metrics_with_thresholds(all_metrics, model, list_of_thresholds)

def calculate_all_metrics(metrics, df, model, threshold):
    TP = len(df.loc[(df["GT"] == 1) & (df[model[1]] > threshold)])
    FP = len(df.loc[(df["GT"] == 0) & (df[model[1]] > threshold)])
    FN = len(df.loc[(df["GT"] == 1) & (df[model[0]] >= 1 - threshold)])
    TN = len(df.loc[(df["GT"] == 0) & (df[model[0]] >= 1 - threshold)])

    if TP + TN + FP + FN == 0: 
        return

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    try:
        precision = TP / (TP + FP) 
    except ZeroDivisionError:
        precision = 0

    try:
        recall = TP / (TP + FN) 
    except ZeroDivisionError:
        recall = 0

    try:
        FPR = FP / (FP + TN)
    except ZeroDivisionError:
        FPR = 0

    b = 1
    try:
        f_scores = (1 + b**2) * precision * recall / (b**2 * (precision + recall))
    except: 
        f_scores = 0

    try:
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    except ZeroDivisionError:
        MCC = 0

    try:
        BA = (TP / (TP + FN) + TN / (TN + FP)) / 2
    except ZeroDivisionError:
        BA = 0

    try:
        Y_J_statistics = TP / (TP + FN) + TN / (TN + FP) - 1
    except ZeroDivisionError:
        Y_J_statistics = 0

    metrics['accuracy'] += [round(accuracy, 4)]
    metrics['precision'] += [round(precision, 4)]
    metrics['recall'] += [round(recall, 4)]
    metrics['f_scores'] += [round(f_scores, 4)]
    metrics['MCC'] += [round(MCC, 4)]
    metrics['BA'] += [round(BA, 4)]
    metrics['Y_J_statistics'] += [round(Y_J_statistics, 4)]
    metrics['FPR'] += [round(FPR, 4)]
    metrics['class_0'].append(TN + FN)
    metrics['class_1'].append(TP + FP)

calculate_metrics_data_frame(data, 0.2)

