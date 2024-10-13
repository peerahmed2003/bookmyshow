from sklearn import  datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import   KNeighborsClassifier
from  sklearn.metrics import accuracy_score

iris =  datasets.load_iris()
X =iris.data
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.2,random_state=42)
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
predictions  = knn.predict(X_test)

correct_predictions=0
wrong_predictions=0
for i in range(len(predictions)):
    if predictions[i]==y_test[i]:
        print(f"correct prediction: actual class-{y_test[i]} prediced class -  {predictions[i]}")
        correct_predictions+=1
    else:
        print(f"wrong prediction: actual class-{y_test[i]} prediced class -  {predictions[i]}")
        wrong_predictions+=1

accuracy = accuracy_score(y_test,predictions)
print(f"Overall accuracy is: {accuracy*100:.2f}%")
print(f"Correct predictions:{correct_predictions}")
print(f"wrong predictions:{wrong_predictions}")