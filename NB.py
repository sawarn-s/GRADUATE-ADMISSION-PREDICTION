import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer, PowerTransformer
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection



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


df.columns = ["No", "GRE", "TOEFL", "UR", "SOP", "LOR", "CGPA", "RES", "CoA",]



X = df[[i for i in list(df.columns) if i != 'CoA']]
y = df['CoA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

# Standardizing the features:
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# chose 0.82 because it is the 3rd quartile for chance of admit
ty_train=[1 if CoA > 0.82 else 0 for CoA in y_train] # learned from internet
ty_train=np.array(ty_train)

ty_test=[1 if CoA > 0.82 else 0 for CoA in y_test] #learned from internet
ty_test=np.array(ty_test)

#nb = GaussianNB()
#nb = MultinomialNB()
#nb = ComplementNB()
#nb = BernoulliNB()
#nb = CategoricalNB()
for model in [GaussianNB, BernoulliNB]:
    print()
    print(model)
    nb = model()
    nb.fit(X_train_std, ty_train)
    y_pred = nb.predict(X_test_std)



    #Model Performance
    # #setting performance parameters
    kfold = model_selection.KFold(n_splits=10)
    #calling the cross validation function
    cv_results = model_selection.cross_val_score(model(), X_train_std, ty_train, cv=kfold, scoring='accuracy')
    #displaying the mean and standard deviation of the prediction
    print("%s: %f %s: (%f)" % ('Naive Bayes accuracy', cv_results.mean(), '\nNaive Bayes StdDev', cv_results.std()))
    print()
    #calling the cross validation function
    cv_results = model_selection.cross_val_score(model(), X_train_std, ty_train, cv=kfold, scoring='precision')
    #displaying the mean and standard deviation of the prediction
    print("%s: %f %s: (%f)" % ('Naive Bayes precision', cv_results.mean(), '\nNaive Bayes StdDev', cv_results.std()))
    print()
    #calling the cross validation function
    cv_results = model_selection.cross_val_score(model(), X_train_std, ty_train, cv=kfold, scoring='recall')
    #displaying the mean and standard deviation of the prediction
    print("%s: %f %s: (%f)" % ('Naive Bayes recall', cv_results.mean(), '\nNaive Bayes StdDev', cv_results.std()))

    print()
    #calling the cross validation function
    cv_results = model_selection.cross_val_score(model(), X_train_std, ty_train, cv=kfold, scoring='f1')
    #displaying the mean and standard deviation of the prediction
    print("%s: %f %s: (%f)" % ('Naive Bayes F1', cv_results.mean(), '\nNaive Bayes StdDev', cv_results.std()))
# Naive Bayes accuracy: 0.926108 
# Naive Bayes StdDev: (0.048931)

# Naive Bayes precision: 0.754275 
# Naive Bayes StdDev: (0.155794)

# Naive Bayes recall: 0.904762 
# Naive Bayes StdDev: (0.145997)

# Naive Bayes F1: 0.817904 
# Naive Bayes StdDev: (0.139074)

X_combined_std = np.vstack((X_train_std[:, 1:3], X_test_std[:, 1:3]))
y_combined = np.hstack((ty_train, ty_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=type(nb)())
plt.savefig("nb.png")
plt.show()
