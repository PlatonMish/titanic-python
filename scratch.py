# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


data = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
test_ids = test["PassengerId"]
def clean(data):
    data = data.drop(["Ticket","Cabin", "Name","PassengerId"], axis=1)

    cols = ["SibSp", "Parch", "Fare", "Age"]
    for col in cols:
        data[col].fillna(data[col].mean(), inplace = True)

    data.Embarked.fillna("U", inplace=True)
    return data

data = clean(data)
test = clean(test

data.head(5)

#from sklearn import preprocessing
le = preprocessing.LabelEncoder()

cols = ["Sex", "Embarked"]

for col in cols:
    data[col] = le.fit_transform(data[col])
    test[col] = le.transform(test[col])
    print(le.classes_)

data.head(5)

#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split

Y = data["Survived"]
X = data.drop ("Survived", axis = 1)

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.2, random_state=43)

clf = LogisticRegression(random_state = 0, max_iter=1000).fit(X_train, y_train)

predictions = clf.predict(X_val)
from sklearn.metrics import accuracy_score
accuracy_score(y_val, predictions)

submission_preds = clf.predict(test)

df = pd.DataFrame({"PassengerId":test_ids.values,
                  "Survived": submission_preds,
                  })


df.to_csv("submission.csv", index = False)