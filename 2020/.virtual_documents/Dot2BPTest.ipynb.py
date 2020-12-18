from PyRNA import dotfile2bp, dot2bp, visualize_structure
import pandas as pd
from sklearn.metrics import recall_score, precision_score, matthews_corrcoef


# Sensitivity, hit rate, recall, or true positive rate: TPR = TP/(TP+FN)
# Specificity or true negative rate: TNR = TN/(TN+FP) 
# Precision or positive predictive value: PPV = TP/(TP+FP)
# Matthews correlation coefficient (MCC): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html#sklearn.metrics.matthews_corrcoef


bp_matrix_ref = dotfile2bp("data/research/reference.ss")
bp_matrix_pre = dotfile2bp("data/research/predicted.ss")


ref = pd.DataFrame.from_records(bp_matrix_ref).values.flatten()
pre = pd.DataFrame.from_records(bp_matrix_pre).values.flatten()
tpr, ppv, mcc = recall_score(y_true = ref, y_pred=pre), precision_score(y_true = ref, y_pred=pre), matthews_corrcoef(y_true = ref, y_pred=pre)
print(tpr, ppv, mcc)


visualize_structure(bp_matrix_ref, label="FSW-reference")
visualize_structure(bp_matrix_pre, label="FSW-predicted")
