import pandas as pd 
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SEED = 20

model = LinearSVC()

uri='https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv'

datas = pd.read_csv(uri)

x = datas[["home", "how_it_works", "contact"]]
y = datas["bought"]

trainX, testX, trainY, testY = train_test_split(x, y, random_state = SEED, test_size = 0.25, stratify = y)

model.fit(trainX, trainY)

predictions = model.predict(testX)

accuracyScore = accuracy_score(testY, predictions)
print("The accuracy was: %.2f " % (accuracyScore * 100))
