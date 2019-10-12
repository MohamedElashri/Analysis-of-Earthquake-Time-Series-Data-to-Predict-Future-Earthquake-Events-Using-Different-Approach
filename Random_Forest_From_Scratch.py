
def decision_tree(X_train,y_train, n_features):
    features = np.random.randint(low = 0, high = X_train.shape[1], size = n_features)
    X_train = X_train[:,features]
        #Rotation forrest
    #pca = PCA(n_components=n_features)
    #pca.fit(X_train.transpose())
    #X_train = pca.components_.transpose()
    clf = tree.DecisionTreeClassifier()
    #clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(X_train,y_train)
    return clf,features


def bootstrapping(data, n_bootstrap):
    bootstrap_indices = np.random.randint(low = 0, high = len(data), size = n_bootstrap)
    bootstrapped = data[bootstrap_indices]
    return bootstrapped

def Random_Forest_Algorithm(X_train, y_train, n_trees, n_bootstrap, n_features):
    forrest = []	
    for i in range(n_trees):
        X_tr = bootstrapping(X_train,n_bootstrap)
        y_tr = bootstrapping(y_train, n_bootstrap)
        tree,features = decision_tree(X_tr, y_tr, n_features)
        forrest.append(tree)
    return forrest,features

#forrest,features = Random_Forest_Algorithm(X_train,y_train, 10, 50, 25)
#print(features)

def Forrest_Predictions(X_test,forrest,features):
    forrest_predictions = {}
    for i in range(len(forrest)):
        column_name = "tree_{}".format(i)
        prediction = forrest[i].predict(X_test[:,features])
        forrest_predictions[column_name] = prediction
    forrest_predictions = pd.DataFrame(forrest_predictions)
    return forrest_predictions




def prediction(X_test, forrest, features):
    forrest_predictions = Forrest_Predictions(X_test, forrest, features)
    predictions = forrest_predictions.sum(axis = 1)

    for i in range(len(predictions)):
        if predictions.loc[i]< 5:
            predictions.loc[i] = 0
        else:
            predictions.loc[i] = 1

    return predictions

#prediction = Final_Prediction(forrest_predictions)

#conf_matrix = confusion_matrix(prediction,y_test)

#print(conf_matrix)

#print(np.trace(conf_matrix)/np.sum(conf_matrix))



