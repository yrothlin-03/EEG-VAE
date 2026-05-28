import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    cohen_kappa_score,
    f1_score,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
)

class Metrics:
    def __init__(self, task: str = "classification", num_classes: int | None = None, average_auc: str = "macro"):
        self.task = task
        self.num_classes = num_classes
        self.average_auc = average_auc
        self.reset()

    def reset(self):
        self._outputs = []
        self._targets = []

    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        self._outputs.append(outputs.detach().cpu())
        self._targets.append(targets.detach().cpu())

    def compute(self) -> dict:
        if len(self._targets) == 0:
            return {}

        outputs = torch.cat(self._outputs, dim=0)
        targets = torch.cat(self._targets, dim=0)

        if self.task == "regression":
            y_true = targets.numpy()
            y_pred = outputs.numpy()
            if y_pred.ndim > 1 and y_pred.shape[-1] == 1:
                y_pred = y_pred.squeeze(-1)
            return {
                "MSE": mean_squared_error(y_true, y_pred),
                "RMSE": mean_squared_error(y_true, y_pred, squared=False),
                "MAE": mean_absolute_error(y_true, y_pred),
                "R2": r2_score(y_true, y_pred),
            }

        logits = outputs
        y_true = targets.long().numpy()

        if logits.ndim == 1:
            logits = logits.unsqueeze(1)

        if self.num_classes is None and logits.shape[1] > 1:
            self.num_classes = logits.shape[1]

        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits[:, 0]).numpy()
            y_pred = (probs >= 0.5).astype(int)
            auroc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan")
            labels = [0, 1]
        else:
            probs = torch.softmax(logits, dim=1).numpy()
            y_pred = probs.argmax(axis=1)
            auroc = float("nan")
            labels = list(range(self.num_classes)) if self.num_classes is not None else None

            try:
                present = np.unique(y_true)
                if len(present) > 1 and self.num_classes is not None:
                    y_true_oh = np.zeros((len(y_true), self.num_classes))
                    y_true_oh[np.arange(len(y_true)), y_true] = 1
                    auroc = roc_auc_score(
                        y_true_oh[:, present],
                        probs[:, present],
                        average=self.average_auc,
                        multi_class="ovr",
                    )
            except Exception:
                auroc = float("nan")

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        return {
            "ACC": accuracy_score(y_true, y_pred),
            "BACC": balanced_accuracy_score(y_true, y_pred),
            "AUROC": auroc,
            "KAPPA": cohen_kappa_score(y_true, y_pred),
            "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "F1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": cm,
        }