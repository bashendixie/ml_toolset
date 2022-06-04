from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
from deepforest import CascadeForestClassifier


# X, y = load_digits(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

X = pd.read_csv('data/train_data.csv')
label = pd.read_csv('data/train_label.csv')
y = label.target
model = CascadeForestClassifier(random_state=1, n_trees=300)
model.fit(X.to_numpy(), y.to_numpy())

# 保存模型
s=pickle.dumps(model)
f=open('deepforest_v3.model', "wb+")
f.write(s)
f.close()

# y_pred = model.predict(X_test)
# acc = accuracy_score(y_test, y_pred) * 100
# print("\nTesting Accuracy: {:.3f} %".format(acc))