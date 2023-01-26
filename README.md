# sklearn
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LogisticRegression, ElasticNet
# from sklearn.preprocessing import MinMaxScaler, RobustScaler
# from sklearn.model_selection import train_test_split, GridSearchCV
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

### 1
# data = pd.read_csv('Rain.csv')
# print(data.head())
# y = data['RainTomorrow'].values
# X = data.drop('RainTomorrow', axis=1).values
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1)
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)
# params = { 'C': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#            'penalty': ['l1','l2','elasticnet']
# }
#
# myModel = LogisticRegression(solver='saga')
# myModel.fit(X_train, y_train)
# print(("Test Score:{}; Train Score:{}").format(myModel.score(X_test,y_test),myModel.score(X_train,y_train)))
# # მცირედი სხვაობაა: 0.814 - 0.810
#
# grid =GridSearchCV(myModel, param_grid=params)
# grid.fit(X_train, y_train)
# print(("Best Params: {}").format(grid.best_params_))
# # საუკეთესო პარამეტრებია: 0.4 და l1

### 2
# data = pd.read_csv('Regression.csv')
# print(data.head())
# y = np.array(data['x']).reshape(-1,1)
# X = np.array(data['y']).reshape(-1,1)
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15, random_state=1)
# scaler = RobustScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)
# myModel = ElasticNet()
# myModel.fit(X_train,y_train)
# print(("Test Score:{}; Train Score:{}").format(myModel.score(X_test,y_test),myModel.score(X_train,y_train)))

### 3
# data = pd.read_csv('Cluster.csv')
# print(data.head())
# myCluster = KMeans(n_clusters=4)
# myCluster.fit(data)
# predicted = myCluster.predict(data)
# score1 = silhouette_score(data, predicted)
# myCluster2 = KMeans(n_clusters=5)
# myCluster2.fit(data)
# predicted2 = myCluster2.predict(data)
# score2 = silhouette_score(data, predicted2)
# print(("Clusters:4, Score: {}; Clusters:5, Score: {}").format(score1, score2))
# # უკეთესია 4 კლასტერი, სქორი არის 0.50
#
# plt.scatter(data['X1'], data['X2'], c=predicted)
# plt.scatter(myCluster.cluster_centers_[:,0], myCluster.cluster_centers_[:,1], s=100, c='black')
# plt.show()


### VARIANT3

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import SelectKBest, chi2, f_classif, SelectPercentile
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LinearRegression
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt

### 1
#
# data = pd.read_csv("heart.csv")
# print(data.head())
# y = data['target'].values
# X = data.drop('target', axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1)
# method = SVC(kernel='poly', degree=3)
# method.fit(X_train, y_train)
# print(("Test Score:{}; Train Score:{}").format(method.score(X_test,y_test),method.score(X_train,y_train)))
#
# # შერჩეული მახასიათებლები
# selector = SelectKBest(score_func=chi2, k=5)
# selector.fit(X, y)
# new_X = X.iloc[:,selector.get_support()].values
# X_train, X_test, y_train, y_test = train_test_split(new_X,y, test_size=0.3, random_state=1)
# method = SVC(kernel='poly', degree=3, C=0.8)
# method.fit(X_train, y_train)
# print(("Test Score:{}; Train Score:{}").format(method.score(X_test,y_test),method.score(X_train,y_train)))

### 2
# data = pd.read_csv("Cars.csv")
# print(data.head())
# scaler = StandardScaler()
# y = data['price'].values
# X = data.drop('price', axis=1)
# X = scaler.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1)
# myModel = LinearRegression()
# myModel.fit(X_train,y_train)
# print(("Test Score:{}; Train Score:{}").format(myModel.score(X_test,y_test),myModel.score(X_train,y_train)))
#
#
# # შემცირებული
# pca = PCA(n_components=9)
# new_X = pca.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(new_X,y, test_size=0.3, random_state=1)
# myModel = LinearRegression()
# myModel.fit(X_train,y_train)
# print(("Test Score:{}; Train Score:{}").format(myModel.score(X_test,y_test),myModel.score(X_train,y_train)))

### 3
# data =pd.read_csv('Cluster_varianti3.csv')
# print(data.head())
#
# myClusters1 = KMeans(n_clusters=2)
# myClusters1.fit(data)
# predicted1 = myClusters1.predict(data)
# score1 = silhouette_score(data, predicted1)
# myClusters2 = KMeans(n_clusters=3)
# myClusters2.fit(data)
# predicted2 = myClusters2.predict(data)
# score2 = silhouette_score(data, predicted2)
# myClusters3 = KMeans(n_clusters=4)
# myClusters3.fit(data)
# predicted3 = myClusters3.predict(data)
# score3 = silhouette_score(data, predicted3)
#
# print(("Clusters:2, Score:{}; Clusters:3, Score:{}; Clusters:4, Score:{}").format(score1, score2, score3))
# # საუკეთესო შემთხვევაა 3 კლასტერი, სქორი გვაქვს: 0.557
#
# plt.scatter(data['X1'], data['X2'], c=predicted2)
# plt.scatter(myClusters2.cluster_centers_[:,0], myClusters2.cluster_centers_[:,1], s=100, c='red')
# plt.show()
#
