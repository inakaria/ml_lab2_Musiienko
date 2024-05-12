import pandas as pd
import matplotlib.pyplot as plt
import math
from copy import deepcopy
import numpy as np
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

def metrics_df(df, thresholds_step, ind):
    list_of_thresholds = []
    for i in range(int(1 / thresholds_step) + 1):
        list_of_thresholds.append(round(i * thresholds_step, 4))

    for model in [('Model_1_0', 'Model_1_1'), ('Model_2_0', 'Model_2_1')]:
        metrics = {'accuracy': [], 
                    'precision': [], 
                    'recall': [], 
                    'f_scores': [], 
                    'MCC': [], 
                    'BA': [], 
                    'Y_J_statistics': [], 
                    'FPR': [], 
                    'class_0': [], 
                    'class_1': []}

        for threshold in list_of_thresholds:
            calculate_metrics(metrics, df, model, threshold)

        for key, value in metrics.items():
            print(f'{key}: {value}')
        
        # 3.b
        show_metrics(metrics, model, list_of_thresholds, ind)

        # 3.d
        show_PRC(metrics['recall'], metrics['precision'], model, ind)
        show_ROC(metrics["FPR"], metrics['recall'], model, ind)


def calculate_metrics(metrics, df, model, threshold):
    TP = len(df.loc[(df["GT"] == 1) & (df[model[1]] > threshold)])
    FP = len(df.loc[(df["GT"] == 0) & (df[model[1]] > threshold)])
    FN = len(df.loc[(df["GT"] == 1) & (df[model[0]] >= 1 - threshold)])
    TN = len(df.loc[(df["GT"] == 0) & (df[model[0]] >= 1 - threshold)])

    if(TP + TN + FP + FN == 0): return

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


print('3b. Збудувати  на  одному  графіку  в  одній  координатній  системі (величина порогу; значення метрики)', 
      'графіки усіх обчислених метрик, відмітивши певним чином максимальне значення кожної з них.')

def show_metrics(metrics, model_name, thresholds, ind):
    new_metrics = deepcopy(metrics)
    new_metrics.pop('FPR')
    new_metrics.pop('class_0')
    new_metrics.pop('class_1')

    plt.figure(figsize=(10, 6))
    for metric_name, values in new_metrics.items():
        plt.plot(thresholds, values, label=metric_name)

    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title(f'Metrics for {model_name[0][:7]}')
    plt.legend(loc='upper right')
    plt.grid()

    plt.savefig(f'{ind}_{model_name[0][:7]}_metrics.png')


print('3d. Збудувати для кожного класифікатору PR-криву та ROC-криву, показавши графічно на них значення оптимального порогу.')

def show_PRC(recall, precision, model_name, ind):
    AUC_PRC = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, marker="o", color='blue', lw=2.5, label=f'PRC curve (area = {AUC_PRC:.3f})')
    plt.plot([0, 1], [0, 1], color='orange', lw=2.5, linestyle='-')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({model_name[0][:7]})')
    plt.legend(loc='lower right')

    plt.savefig(f'{ind}_{model_name[0][:7]}_PRC.png')


def show_ROC(FPR, TPR, model_name, ind):
    AUC_PRC = auc(FPR, TPR)

    plt.figure()
    plt.plot(FPR, TPR, marker="o", color='blue', lw=2.5, label=f'ROC curve (area = {AUC_PRC:.3f}')
    plt.plot([1, 0], [0, 1], color='orange', lw=2.5, linestyle='-')
    plt.plot([0, 1], [0, 1], color='green', lw=2, linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic ({model_name[0][:7]})')
    plt.legend(loc='upper right')

    plt.savefig(f'{ind}_{model_name[0][:7]}_ROC.png')

metrics_df(data, 0.2, 1) # Завдання 3


print('5. Створити  новий  набір  даних,  прибравши  з  початкового  набору  (50 + 5К)% об’єктів класу 1,',
      'вибраних випадковим чином. Параметр К представляє собою залишок від ділення дня народження студента на 9',
      'та  має  визначатися  в  програмі  на  основі  дати  народження  студента,',
      'яка задана в програмі у вигляді текстової змінної формату ‘DD-MM’.')
def calculate_K(birthdate):
    day, month = birthdate.split('-')
    K = (int(day) % 9)
    return K

birthdate = '10-08'
K = calculate_K(birthdate)
print("K =", K)

class_1_count = data['GT'].sum()
percent_remove = 50 + 5 * K
objects_remove = int(class_1_count * percent_remove / 100)
class_1_indices = data[data['GT'] == 1].index
objects_remove_indices = np.random.choice(class_1_indices, size=objects_remove, replace=False)
new_data = data.drop(objects_remove_indices)


print('6. Вивести відсоток видалених об’єктів класу 1 та кількість елементів кожного класу після видалення.')

print("Кількість об'єктів класу 1 в початковому наборі даних:", class_1_count)
print("Кількість об'єктів класу 0 в початковому наборі даних:", len(data) - class_1_count)
new_class_1_count = new_data['GT'].sum()

print(f"Відсоток видалених об'єктів класу 1: {percent_remove}%")
print("Кількість об'єктів класу 1 після видалення:", new_class_1_count)
print("Кількість об'єктів класу 0 після видалення:", len(new_data) - new_class_1_count)


print('7. Виконати дії п.3 для нового набору даних.')
metrics_df(new_data, 0.2, 2)  


print('8-9. Визначити кращу модель. Пояснити вплив незбалансованості набору даних на прийняте рішення.')
new_class_counts = new_data['GT'].value_counts()
print(new_class_counts)
