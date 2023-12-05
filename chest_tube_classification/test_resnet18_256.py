from dataset import CustomDataset, CustomDatasetWeek, test_transform
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
import os
import itertools

from models import SmallCNN, get_custom_resnet18
from constants import CHECKPOINT_DIR


def plot_confusion_matrix(cm, classes, model_name):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Displaying the text in the middle of the cells
    thresh = cm.max() / 2.  # This is for choosing the text color
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")  # Setting the text color based on the cell color

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'{model_name}_eval/confusion_matrix.png')
    plt.close()


def plot_roc_curve(y_test, y_score, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'{model_name}_eval/roc_curve.png')
    plt.close()


def plot_precision_recall_curve(y_test, y_score, model_name):
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    average_precision = average_precision_score(y_test, y_score)

    baseline = sum(y_test) / len(y_test)

    plt.figure()
    plt.plot(recall, precision, label=f'Precision-Recall curve (area = {average_precision:.2f})')
    plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline (precision = {baseline:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend(loc="upper right")
    plt.savefig(f'{model_name}_eval/precision_recall_curve.png')
    plt.close()


def test(model, dataloader, criterion, device, model_name):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_scores = []

    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            probabilities = softmax(outputs)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probabilities[:, 1].cpu().numpy())

    accuracy = 100. * correct / total
    print(f"Test loss: {test_loss / len(dataloader)}, Accuracy: {accuracy}%")
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    print(f"Test loss: {test_loss / len(dataloader)}, Accuracy: {accuracy}%, F1 score: {f1_score}")
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    if not os.path.exists(f'{model_name}_eval'):
        os.makedirs(f'{model_name}_eval')

    y_test = all_labels
    y_score = all_scores

    plot_confusion_matrix(cm, [0, 1], model_name) # You might need to adjust [0, 1] based on your classes
    plot_roc_curve(y_test, y_score, model_name=model_name)
    plot_precision_recall_curve(y_test, y_score, model_name=model_name)
    
    return test_loss / len(dataloader), accuracy


if __name__ == '__main__':
    model_name = 'resnet18_256_hist_eq0.0001_64'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64

    test_dataset = CustomDataset('test_org_filtered', augmentations=test_transform, kind=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = get_custom_resnet18()
    model_path = CHECKPOINT_DIR + '/' + model_name + '/' + 'best_model.pth'
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    test(model, test_loader, criterion, device, model_name)
