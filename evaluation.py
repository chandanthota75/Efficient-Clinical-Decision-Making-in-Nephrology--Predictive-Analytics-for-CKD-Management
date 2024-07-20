import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import csv


def eval_function_with_roc_plot(models, model_names, legend_names, colors_names, test_data, test_labels):
    curves = []

    for model_name in model_names:
        name = legend_names[model_names.index(model_name)]
        model = models[model_name]

        if name == "SVM":
            model_prediction = model.predict(test_data)
            model_proba = model.decision_function(test_data)
        else:
            model_prediction = model.predict(test_data)
            model_proba = model.predict_proba(test_data)[:, 1]

        model_cm = confusion_matrix(test_labels, model_prediction)
        tn, fp, fn, tp = model_cm.ravel()

        model_accuracy = accuracy_score(test_labels, model_prediction)
        model_misclassification = 1 - model_accuracy
        model_precision = precision_score(test_labels, model_prediction)
        model_recall = recall_score(test_labels, model_prediction)
        model_f1 = f1_score(test_labels, model_prediction)
        model_specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
        model_mcc = matthews_corrcoef(test_labels, model_prediction)

        # ROC Curve and AUC
        model_fpr, model_tpr, _ = roc_curve(test_labels, model_proba)
        model_auc = auc(model_fpr, model_tpr)

        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(test_labels, model_proba)
        model_pr_auc = average_precision_score(test_labels, model_proba)

        # Print metrics
        print(f"Confusion matrix of the model {name}:\n{model_cm}")
        print(f'True Negative of the model {name} -> {tn}')
        print(f'False Positive of the model {name} -> {fp}')
        print(f'False Negative of the model {name} -> {fn}')
        print(f'True Positive of the model {name} -> {tp}')
        print(f"Accuracy of the model {name} -> {model_accuracy}")
        print(f"Misclassification rate of the model {name} -> {model_misclassification}")
        print(f"Precision of the model {name} -> {model_precision}")
        print(f"Recall of the model {name} -> {model_recall}")
        print(f"F1 Measure of the model {name} -> {model_f1}")
        print(f"Specificity of the model {name} -> {model_specificity}")
        print(f"MCC of the model {name} -> {model_mcc}")
        print(f"AUC of the model {name} -> {model_auc}")
        print(f"Precision-Recall AUC of the model {name} -> {model_pr_auc}")
        print("\n")

        # Plot ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(model_fpr, model_tpr, color=colors_names[model_names.index(model_name)], lw=3,
                 label=f'{name} (AUC = {model_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='black', linestyle='--', lw=2, alpha=0.5)
        plt.legend(loc='lower right', fontsize=12)
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'Receiver Operating Characteristic for {name}', fontsize=14)
        plt.show()

        # Save metrics to CSV
        csv_filename = f"models/{name}_metrics.csv"
        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Metric', 'Value'])
            csv_writer.writerow(['True Negative', tn])
            csv_writer.writerow(['False Positive', fp])
            csv_writer.writerow(['False Negative', fn])
            csv_writer.writerow(['True Positive', tp])
            csv_writer.writerow(['Accuracy', model_accuracy])
            csv_writer.writerow(['Misclassification rate', model_misclassification])
            csv_writer.writerow(['Precision', model_precision])
            csv_writer.writerow(['Recall', model_recall])
            csv_writer.writerow(['F1 Measure', model_f1])
            csv_writer.writerow(['Specificity', model_specificity])
            csv_writer.writerow(['MCC', model_mcc])
            csv_writer.writerow(['AUC', model_auc])
            csv_writer.writerow(['Precision-Recall AUC', model_pr_auc])

        # Append to respective lists
        if name in legend_names:
            curves.append([model_fpr, model_tpr, colors_names[model_names.index(model_name)], name, model_auc])

    return curves


def plot_roc_curves(roc_curves, names):
    for i in range(len(roc_curves)):
        plt.figure(figsize=(12, 7))
        for j in range(len(roc_curves[i])):
            plt.plot(roc_curves[i][j][0], roc_curves[i][j][1], color=roc_curves[i][j][2], lw=2,
                     label=f'Model {roc_curves[i][j][3]} with AUC = {round(roc_curves[i][j][4], 4)}')

        plt.plot([0, 1], [0, 1], color='black', linestyle='--', lw=2, alpha=0.5)

        # Customize legend position
        plt.legend(loc='lower right', fontsize=10.5)

        # Add labels and title
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'Receiver Operating Characteristic for all the {names[i]} learning models', fontsize=14)

        # Show the plot
        plt.show()
