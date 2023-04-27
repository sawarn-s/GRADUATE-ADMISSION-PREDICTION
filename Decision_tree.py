from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    classifier.fit(X, y)
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')





#read the data into a pandas dataframe
df = pd.read_csv('Admission_Predict_Ver1.1.csv')
df.dropna(inplace=True)


df.columns = ["No", "GRE", "TOEFL", "UR", "SOP", "LOR", "CGPA", "RES", "CoA"]


#print(df.describe())
X = df[['GRE', 'TOEFL', 'CGPA']]
y = df['CoA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

# chose 0.82 because it is the 3rd quartile for chance of admit
ty_train=[1 if CoA > 0.82 else 0 for CoA in y_train] # learned from internet
ty_train=np.array(ty_train)

ty_test=[1 if CoA > 0.82 else 0 for CoA in y_test] #learned from internet
ty_test=np.array(ty_test)

tree_model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=1)
tree_model.fit(X_train, ty_train)
dt_pred = tree_model.predict(X_test)

dot_data = export_graphviz(tree_model,
                           filled=True, 
                           rounded=True,
                           class_names=['Rejected', 'Admitted'],
                           feature_names=['GRE', 'TOEFL', 'CGPA'],
                           out_file=None) 
graph = graph_from_dot_data(dot_data)

graph.write_png("tree.png")

X_combined_std = np.vstack((X_train[['CGPA', 'TOEFL']], X_test[['CGPA', 'TOEFL']]))
y_combined = np.hstack((ty_train, ty_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=1))
plt.savefig("decision_tree.png")
plt.show()

print("Decision Tree Accuracy: %.3f" % accuracy_score(ty_test, dt_pred))
print("Decision Tree F1-Score: %.3f" % f1_score(ty_test, dt_pred))
print("Decision Tree Precision: %.3f" % precision_score(ty_test, dt_pred))
print("Decision Tree Recall: %.3f" % recall_score(ty_test, dt_pred))
# Decision Tree Accuracy: 0.931
# Decision Tree F1-Score: 0.898
# Decision Tree Precision: 0.957
# Decision Tree Recall: 0.846
