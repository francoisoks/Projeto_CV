from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
import numpy as np

class Metrics:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.targets = []
        self.predictions = []
        self.losses = []

    def add_predictions(self, targets, predictions, loss=None):
        self.targets.extend(targets)
        self.predictions.extend(predictions)
        if loss is not None:
            self.losses.append(loss)

    def calc_metrics(self):
        balanced_acc = balanced_accuracy_score(self.targets, self.predictions)
        f1 = f1_score(self.targets, self.predictions, average='weighted', zero_division=0)
        recall = recall_score(self.targets, self.predictions, average='weighted', zero_division=0)
        avg_loss = np.mean(self.losses) if self.losses else 0.0

        return {
            'balanced_acc': balanced_acc,
            'f1_score': f1,
            'recall': recall,
            'loss': avg_loss,
        }
