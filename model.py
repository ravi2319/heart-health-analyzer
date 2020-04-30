import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv('heart.csv')

X = df.drop(['target'], axis=1).values
y = df.target.values

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)

X_test_scaled = scaler.fit_transform(X_test)
X_test = pd.DataFrame(X_test_scaled)

knn = KNeighborsClassifier()

params = {'n_neighbors': list(range(1, 20)),
          }

model = GridSearchCV(knn, params, cv=3)

model.fit(X_train, y_train)
print(model.best_params_)

z = model.predict(X_test)

test_score = accuracy_score(y_test, z)
train_score = accuracy_score(y_train, model.predict(X_train))
print(test_score)
print(train_score)

#z = np.array([54, 1, 2, 135, 200, 1, 0, 140, 0, 1, 2, 0, 2]).reshape(-1,1)

pickle.dump(model, open('model.pkl','wb'))

#print(model.predict([[54, 1, 2, 135, 200, 1, 0, 140, 0, 1, 2, 0, 2]]))

