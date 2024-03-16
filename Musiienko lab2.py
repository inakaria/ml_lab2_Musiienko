import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, precision_recall_curve, roc_curve
import numpy as np
from matplotlib.cm import get_cmap


print('1. Відкрити та зчитати файл з даними.')
data = pd.read_csv('KM-12-2.csv')

print('2. Визначити збалансованість набору даних. Вивести кількість об’єктів кожного класу.')
class_counts = data['GT'].value_counts()
print(class_counts) # Кількість об'єктів кожного класу однакова, ми можемо вважати набір даних збалансованим.


print('3а. Обчислити всі метрики (Accuracy, Precision, Recall, F-Scores, Matthews Correlation Coefficient,', 
      'Balanced Accuracy, Youden’s J statistics,  Area  Under  Curve  for  Precision-Recall  Curve,', 
      'Area  Under  Curve  for  Receiver  Operation  Curve)  для  кожної  моделі  при  різних значеннях порогу класифікатора', 
      '(крок зміни порогу 0.2).')

def compute_metrics(data, threshold_step=0.2):
    metrics = {}

    for model in data.columns[1:]:
        metrics[model] = {}
        for threshold in np.arange(0, 1, threshold_step):
            predicted_class = (data[model] >= threshold).astype(int)
            accuracy = accuracy_score(data['GT'], predicted_class)
            precision = precision_score(data['GT'], predicted_class)
            recall = recall_score(data['GT'], predicted_class)
            f1 = f1_score(data['GT'], predicted_class)
            mcc = matthews_corrcoef(data['GT'], predicted_class)
            balanced_accuracy = balanced_accuracy_score(data['GT'], predicted_class)
            
            fpr, tpr, _ = roc_curve(data['GT'], data[model])
            auc_roc = roc_auc_score(data['GT'], data[model])
            
            precision_, recall_, _ = precision_recall_curve(data['GT'], data[model])
            auc_pr = np.trapz(recall_, precision_)

            metrics[model][threshold] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Matthews Correlation Coefficient': mcc,
                'Balanced Accuracy': balanced_accuracy,
                'Area Under Curve (ROC)': auc_roc,
                'Area Under Curve (PR)': auc_pr
            }

    return metrics

metrics = compute_metrics(data)


print('3b. Збудувати  на  одному  графіку  в  одній  координатній  системі  ( величина порогу; значення метрики)', 
      'графіки усіх обчислених метрик, відмітивши певним чином максимальне значення кожної з них')

def plot_metrics(metrics):
    plt.figure(figsize=(12, 8))

    # Використовуємо колірну мапу для генерації кольорів для кожної моделі
    cmap = get_cmap('tab10')

    for i, (model, thresholds) in enumerate(metrics.items()):
        model_color = cmap(i)  # Колір для поточної моделі
        for metric, values in thresholds.items():
            x = list(values.keys())
            y = list(values.values())
            plt.plot(x, y, label=f"{model} - {metric}", color=model_color)

    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('Metrics vs Threshold')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Побудова графіків метрик
plot_metrics(metrics)


print('4. Зробити висновки щодо якості моделей, визначити кращу модель.')
print('Найвищі значення метрик мають моделі Model_1_1 та Model_2_1, тому їх можна вважати якісними.',
      'Загалом модель Model_1_1 має вищі показники за Model_2_1 (окрім метрики Recall), а отже вважається кращою.',
      'Найнижчі значення метрик має модель Model_1_0.')


print('5. Створити  новий  набір  даних,  прибравши  з  початкового  набору  (50 + 5К)% об’єктів класу 1,',
      'вибраних випадковим чином. Параметр К представляє собою залишок від ділення дня народження студента на 9',
      'та  має  визначатися  в  програмі  на  основі  дати  народження  студента,',
      'яка задана в програмі у вигляді текстової змінної формату ‘DD-MM’.')
def calculate_K(birthdate):
    day, month = map(int, birthdate.split('-'))
    K = (day % 9)
    return K

birthdate = '10-08'
K = calculate_K(birthdate)

class_1_count = data['GT'].sum()
percent_to_remove = 50 + 5 * K
objects_to_remove = int(class_1_count * percent_to_remove / 100)
class_1_indices = data[data['GT'] == 1].index
objects_to_remove_indices = np.random.choice(class_1_indices, size=objects_to_remove, replace=False)
new_data = data.drop(objects_to_remove_indices)

print("Кількість об'єктів класу 1 в початковому наборі даних:", class_1_count)
print("Кількість об'єктів класу 1 у новому наборі даних:", new_data['GT'].sum())


print('6. Вивести відсоток видалених об’єктів класу 1 та кількість елементів кожного класу після видалення.')
removed_percentage = (objects_to_remove / class_1_count) * 100

# Кількість об'єктів класу 1 та 0 після видалення
new_class_1_count = new_data['GT'].sum()
new_class_0_count = len(new_data) - new_class_1_count

print("Відсоток видалених об'єктів класу 1: {:.2f}%".format(removed_percentage))
print("Кількість об'єктів класу 1 після видалення:", new_class_1_count)
print("Кількість об'єктів класу 0 після видалення:", new_class_0_count)


print('7. Виконати дії п.3 для нового набору даних.')
new_metrics = compute_metrics(new_data) # 3a
plot_metrics(new_metrics) # 3b