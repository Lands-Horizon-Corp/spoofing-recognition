from __future__ import annotations

import pandera as pa
from pandera.typing import DataFrame
from torchmetrics import ConfusionMatrix


class ModelEval(pa.DataFrameModel):
    preds: float
    label: int


@pa.check_types
def calculate_confusion_matrix(model_eval_results: DataFrame[ModelEval]):
    """
    Calculate the confusion matrix for binary classification.

    Args:
        model_eval_result (ModelEvalResult): An instance containing predicted and true labels.

    Returns:
        Tensor: The calculated confusion matrix.
    """
    confusion_matrix = ConfusionMatrix(task='binary')

    confusion_matrix.update(model_eval_results.preds, model_eval_results.label)
    fig, ax = confusion_matrix.plot()
    return fig
