from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.patches as patches  # extra code – for the curved arrow
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from common import dataset_path, image_path
from common import save_fig
from common import plot_digit

# 支持向量: 离超平面最近的点
IMAGES_PATH = image_path()/"classification"
if IMAGES_PATH is None:
    raise ValueError(
        "IMAGES_PATH is None. Please check the image_path function.")
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

# ==== Dataset ====

mnist = fetch_openml('mnist_784', data_home=dataset_path(),
                     as_frame=False, parser='auto')
print("mnist:", mnist.keys())
X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# True for 5, False for other digits
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

# n_samples, n_features 样本 特征
some_digit = X[0]
# 读取矩阵的长度
shape = some_digit.shape
plot_digit(some_digit)
save_fig("some_digit_plot", IMAGES_PATH)
plt.show()

# ==== Model ====

# SGD 随机梯度下降
# 默认拟合使用 linear support vector machine(SVM)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

predict = sgd_clf.predict([some_digit])
print("predict:", predict)
print("some_digit len:",  shape, "Target value:", y[0])

# ==== Cross-Validation(Measuring Accuracy) 交叉验证 ====

sgd_score = cross_val_score(
    sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print("cross_val_score:", sgd_score)

# add shuffle=True (打乱) if the dataset is not
skfolds = StratifiedKFold(n_splits=3)
# already shuffled

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print("score:", n_correct / len(y_pred))

# This is simply because only about 10% of the images are 5s,
# so if you always guess that an image is not a 5, you will be right about 90% of the time.
dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)
print(any(dummy_clf.predict(X_train)))
dummy_score = cross_val_score(
    dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print("dummy_score:", dummy_score)

# ==== Confusion Matrix 混淆矩阵 ====
# The general idea of a confusion matrix is to count the number of times
# instances of class A are classified as class B, for all A/B pairs.

# cross_val_predict
# instead of returning the evaluation scores, it returns the predictions made on each test fold.
# This means that you get a clean prediction for each instance in the training set
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
cm = confusion_matrix(y_train_5, y_train_pred)

# Precision and Recall

# precision = TP TP+FP
# recall = TP TP+FN
precision = precision_score(y_train_5, y_train_pred)
# == 3530 / (687 + 3530)
print("precision:", precision)
recall = recall_score(y_train_5, y_train_pred)
# == 3530 / (1891 + 3530)
print("recall:", recall)
f1 = f1_score(y_train_5, y_train_pred)
print("f1_score:", f1)
print("cm[1, 1] / (cm[1, 0] + cm[1, 1]):", cm[1, 1] / (cm[1, 0] + cm[1, 1]))

# Precision/Recall Trade-off
y_scores = sgd_clf.decision_function([some_digit])
print("y_scores:", y_scores)
threshold = 0
y_some_digit_pred = (y_scores > threshold)
print("y_some_digit_pred:", y_some_digit_pred)

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]
threshold_for_90_precision
print("threshold_for_90_precision:", threshold_for_90_precision)

y_train_pred_90 = (y_scores >= threshold_for_90_precision)
precision_score(y_train_5, y_train_pred_90)
recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
print("recall_at_90_precision:", recall_at_90_precision)

# ==== The ROC(receiver operating characteristic) Curve ====

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting
plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")

# extra code – just beautifies and saves Figure 3–7
plt.gca().add_patch(patches.FancyArrowPatch(
    (0.20, 0.89), (0.07, 0.70),
    connectionstyle="arc3,rad=.4",
    arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
    color="#444444"))
plt.text(0.12, 0.71, "Higher\nthreshold", color="#333333")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.grid()
plt.axis([0, 1, 0, 1])
plt.legend(loc="lower right", fontsize=13)
save_fig("roc_curve_plot", IMAGES_PATH)
plt.show()
# the area under the curve (AUC)
roc_auc_score = roc_auc_score(y_train_5, y_scores)
print("roc_auc_score:", roc_auc_score)

# 随机森林
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")
# Not in the code
idx_50_to_60 = (y_probas_forest[:, 1] > 0.50) & (y_probas_forest[:, 1] < 0.60)
print(f"{(y_train_5[idx_50_to_60]).sum() / idx_50_to_60.sum():.1%}")
y_scores_forest = y_probas_forest[:, 1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(
    y_train_5, y_scores_forest)
plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting
plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2,
         label="Random Forest")
plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])
plt.grid()
plt.legend(loc="lower left")
save_fig("pr_curve_comparison_plot", IMAGES_PATH)
plt.show()

y_train_pred_forest = y_probas_forest[:, 1] >= 0.5  # positive proba ≥ 50%
precision_score(y_train_5, y_train_pred_forest)
recall_score(y_train_5, y_train_pred_forest)
# roc_auc_score(y_train_5, y_scores_forest)
# f1_score(y_train_5, y_train_pred_forest)

# Multiclass Classification

# Support Vector Classification
# SVMs do not scale well to large datasets, so let's only train on the first 2,000 instances, or else this section will take a very long time to run:
svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])  # y_train, not y_train_5
predict_result = svm_clf.predict([some_digit])
print("predict:", predict_result)

ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])
predict_result = ovr_clf.predict([some_digit])
print("predict:", predict_result)

# Error Analysis

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
plt.rc('font', size=9)  # extra code – make the text smaller
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
plt.show()

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
plt.rc('font', size=9)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=axs[0])
axs[0].set_title("Confusion matrix")
plt.rc('font', size=10)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=axs[1],
                                        normalize="true", values_format=".0%")
axs[1].set_title("CM normalized by row")
save_fig("confusion_matrix_plot_1", IMAGES_PATH)
plt.show()

# Multilabel Classification

y_train_large = (y_train >= '7')
y_train_odd = (y_train.astype('int8') % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
knn_predict = knn_clf.predict([some_digit])
print("knn_predict:", knn_predict)


# Multioutput Classification
np.random.seed(42)  # to make this code example reproducible
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test
plt.subplot(121)
plot_digit(X_test_mod[0])
plt.subplot(122)
plot_digit(y_test_mod[0])
save_fig("noisy_digit_example_plot", IMAGES_PATH)
plt.show()

plt.subplot(121)
plot_digit(X_test_mod[0])
plt.subplot(122)
plot_digit(y_test_mod[0])
save_fig("noisy_digit_example_plot", IMAGES_PATH)
plt.show()

print("DONE.")
