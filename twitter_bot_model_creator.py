import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

pd.set_option('display.max_rows', None)
relevant_data = ["followers_count","friends_count", "listedcount", "favourites_count", "verified","bot"]
data = pd.read_csv('bot_data.csv')
data = data[relevant_data]
X = data[["followers_count","friends_count", "listedcount", "favourites_count","verified"]]
Y = data[["bot"]]
data.head()


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
X_train.shape, Y_train.shape, X_test.shape, Y_test.shape


model = DecisionTreeClassifier(random_state=0)

model.fit(X_train, Y_train)

with open('twitter_decision_tree_model.pkl', 'wb') as model_file:
	pickle.dump(model,model_file)
cross_val_score(model, X_train, Y_train, cv=5)

print('Accuracy of linear SVC on training set: {:.2f}'.format(model.score(X_train, Y_train)))

print('Accuracy of linear SVC on test set: {:.2f}'.format(model.score(X_test, Y_test)))


fig = plt.figure(figsize=(30,50))
_ = tree.plot_tree(model, feature_names=relevant_data, class_names=["bot","human"], filled=True)


# this code is adopted from this example:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

np.set_printoptions(precision=2)
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(model, X_test, Y_test,
                                 display_labels=["bot","human"],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    # print(disp.confusion_matrix)

plt.show()
