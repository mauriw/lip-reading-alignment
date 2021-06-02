from google.colab import drive
drive.mount('/content/drive')

import csv
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
def get_datapoints(folder):
  ans = []
  with open(f"/content/drive/Shareddrives/CS229 Project/Final project/new_datasets/{folder}/filename_embedding_label.csv", "r") as csv4:
    sw = csv.reader(csv4, delimiter=',')
    for i in sw:
      ans.append(i)
  x = []
  y = []
  for i in ans:
    x.append(list(map(float, i[1:-1])))
    y.append(int(i[-1]))
  return x,y
x,y = get_datapoints('pNttjJUtkA4')
clf = make_pipeline(StandardScaler(), SVC(gamma='auto', class_weight="balanced"))
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
clf.fit(x_train, y_train)

y_eval = clf.predict(x_test)
print(np.mean(y_eval == y_test))
