# Importing KNN
from sklearn.neighbors import KNeighborsRegressor  # Notice here we are taking regressor because we are predicting a continuous variable
     from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()scaled = scaler.fit_transform(X)
model_knn = KNeighborsRegressor()
xtrain, xtest, ytrain, ytest = train_test_split(
    X, Y, random_state=42, test_size=0.3, shuffle=True)
     model_knn.fit(xtrain, ytrain)
  knn_pred_test = model_knn.predict(xtest)
  knn_pred_train = model_knn.predict(xtrain)
  knn_r2_train = r2_score(ytrain, knn_pred_train)
knn_r2_test = r2_score(ytest, knn_pred_test)
knn_r2_train, knn_r2_test
train_r2_scores = []
test_r2_scores = []

for n in range(1, 30):
    # Initializing KNN
    model_knn = KNeighborsRegressor(n_neighbors=n)
    
    # Fitting the data and taking predictions for both training and testing dataset
    model_knn.fit(xtrain, ytrain)
    knn_pred_train, knn_pred_test = model_knn.predict(xtrain), model_knn.predict(xtest)
    
    # Storing the R-Squared Coefficient into list
    train_r2_scores.append(r2_score(ytrain, knn_pred_train))
    test_r2_scores.append(r2_score(ytest, knn_pred_test))
    plt.figure(figsize=(15, 7))
sns.lineplot(x=range(1, 30), y=train_r2_scores, label='Training')
sns.lineplot(x=range(1, 30), y=test_r2_scores, label='Testing')
plt.xticks(range(1, 30))
#Checking the degree of polynomial from 1 to 6
train_r2_scores = []
test_r2_scores = []
for degree in range(1, 7):
    # Converting the features to polynomial
    model_poly = PolynomialFeatures(degree=degree)
    xtrain_poly, xtest_poly = model_poly.fit_transform(xtrain), model_poly.fit_transform(xtest)
    
    # Initializing KNN
    model_knn = KNeighborsRegressor(n_neighbors=11)
    model_knn.fit(xtrain_poly, ytrain)
    
    # Getting the training score in the list
    train_r2_scores.append(r2_score(ytrain, model_knn.predict(xtrain_poly)))
    test_r2_scores.append(r2_score(ytest, model_knn.predict(xtest_poly)))

# Displaying the model complexity
plt.figure(figsize=(10, 7))
sns.lineplot(x=range(1, 7), y=train_r2_scores, label='Training')
sns.lineplot(x=range(1, 7), y=test_r2_scores, label='Testing')
plt.xlabel('Degrees', fontsize=15)
plt.ylabel('R-Squared Coef.', fontsize=15)
plt.title('Model Complexity', fontsize=15)
plt.yticks([0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82])
plt.show()
list(zip(train_r2_scores, test_r2_scores))
     
