from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json

# =========================================================================================
# load train dataset js file
with open('./inputs/dataset_train.js', 'r') as file:
    js_data = json.loads(file.read())

train_data = []

for i in js_data:
    train_data.append(list(i.values()))

# =========================================================================================
# generate label according to absolute right rule
train_label = []

for i in js_data:
    if i["number_of_raise"] <= 2 and i["shopping_frequency"] <= 5 \
            and i["income"] >= 35000 and i["arrears"] <= 1:
        if i["average_cost"] <= 30000:
            train_label.append(1)
        elif i["installment"] <= 15:
            train_label.append(1)
        elif i["passive_income"] >= i["income"] / 5:
            train_label.append(1)
        elif i["deposit"] >= 20000:
            train_label.append(1)
        else:
            train_label.append(0)
    else:
        train_label.append(0)

# =========================================================================================
# load test dataset js file
# you can open test data with noise or without noise (default open without noise)

# with open('./inputs/dataset_test_noise.js') as file:
with open('./inputs/dataset_test_noise.js') as file:
    js_test_data = json.loads(file.read())

test_data = []

for i in js_test_data:
    test_data.append(list(i.values()))

# =========================================================================================
# generate label according to absolute right rule
test_label = []

for i in js_test_data:
    if i["number_of_raise"] <= 2 and i["shopping_frequency"] <= 5 \
            and i["income"] >= 35000 and i["arrears"] <= 1:
        if i["average_cost"] <= 30000:
            test_label.append(1)
        elif i["installment"] <= 15:
            test_label.append(1)
        elif i["passive_income"] >= i["income"] / 5:
            test_label.append(1)
        elif i["deposit"] >= 20000:
            test_label.append(1)
        else:
            test_label.append(0)
    else:
        test_label.append(0)

# =========================================================================================
def calculate_precision(predicted_label, ground_truth):
    predict_1s = 0
    truly_1s = 0
    predict_0s = 0
    truly_0s = 0
    # count total that predict 1
    # count total that's truly 1
    for ind, lb in enumerate(predicted_label):
        if lb:
            predict_1s += 1
            if lb == ground_truth[ind]:
                truly_1s += 1
        else:
            predict_0s += 1
            if lb == ground_truth[ind]:
                truly_0s += 1
    return truly_0s / predict_0s, truly_1s / predict_1s


def calculate_recall(predicted_label, ground_truth):
    label_1s = 0
    truly_1s = 0
    label_0s = 0
    truly_0s = 0
    # count total that label 1
    # count total that's truly 1
    for ind, lb in enumerate(ground_truth):
        if lb:
            label_1s += 1
            if lb == predicted_label[ind]:
                truly_1s += 1
        else:
            label_0s += 1
            if lb == predicted_label[ind]:
                truly_0s += 1

    return truly_0s / label_0s, truly_1s / label_1s


def evaluate(model, data, ground_truth, model_name):
    prediction = model.predict(data)
    precision0, precision1 = calculate_precision(prediction, ground_truth)
    recall0, recall1 = calculate_recall(prediction, ground_truth)
    print(model_name)
    print(
        f"predict for 0s -> precision: {round(precision0 * 100, 2)}, recall: {round(recall0 * 100, 2)}, f1-score: {round(2 * (precision0 * recall0) / (precision0 + recall0) * 100, 2)}")
    print(
        f"predict for 1s -> precision: {round(precision1 * 100, 2)}, recall: {round(recall1 * 100, 2)}, f1-score: {round(2 * (precision1 * recall1) / (precision1 + recall1) * 100, 2)}")
    print()


def draw_confusion(model, name, title):
    disp = ConfusionMatrixDisplay.from_estimator(model, train_data, train_label, cmap="binary")
    plt.title(title)
    disp.figure_.savefig(f'./confusion figure/confusion matrix_{name}.png')


def draw_confusion_noise(model, name, title):
    disp = ConfusionMatrixDisplay.from_estimator(model, test_data, test_label, cmap="binary")
    plt.title(title)
    disp.figure_.savefig(f'./confusion figure/confusion matrix_{name}_Noise.png')


if __name__ == "__main__":
# =========================================================================================
    # decision tree train
    decision_tree = tree.DecisionTreeClassifier()
    decision_tree = decision_tree.fit(train_data, train_label)

    # evaluate
    evaluate(decision_tree, train_data, train_label, "Decision Tree")
    decision_tree_acc = decision_tree.score(train_data, train_label)

    # output confusion matrix image
    # draw_confusion(decision_tree, "DT", "Decision Tree")
    # draw_confusion_noise(decision_tree, "DT", "Decision Tree (Noise)")

# =========================================================================================
    # K-Neighbors train
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_data, train_label)

    # evaluate
    evaluate(neigh, train_data, train_label, "K-Neighbors")
    neigh_acc = neigh.score(train_data, train_label)

    # output confusion matrix image
    # draw_confusion(neigh, "KNN", "K-Neighbors")
    # draw_confusion_noise(neigh, "KNN", "K-Neighbors (Noise)")

# =========================================================================================
    # Logistic Regression train
    logistic_regression = LogisticRegression(random_state=0, max_iter=150).fit(train_data, train_label)

    # evaluate
    evaluate(logistic_regression, test_data, test_label, "Logistic Regression")
    logistic_acc = logistic_regression.score(train_data, train_label)

    # output confusion matrix image
    # draw_confusion(logistic_regression, "LR", "Logistic Regression")
    # draw_confusion_noise(logistic_regression, "LR", "Logistic Regression (Noise)")
# =========================================================================================
    # Gaussian Naive Bayes train
    gaussian_naive_bayes = GaussianNB()
    gaussian_naive_bayes.fit(train_data, train_label)

    # evaluate
    evaluate(gaussian_naive_bayes, train_data, train_label, "Gaussian Naive Bayes")
    gnb_acc = gaussian_naive_bayes.score(train_data, train_label)

    # output confusion matrix image
    # draw_confusion(gaussian_naive_bayes, "GNB", "Gaussian Naive Bayes")
    # draw_confusion_noise(gaussian_naive_bayes, "GNB", "Gaussian Naive Bayes (Noise)")
# =========================================================================================
    # output decision tree image
    # tree.plot_tree(decision_tree, filled=True)
    # plt.savefig('tree.png', format='png')
