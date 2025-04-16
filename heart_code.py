import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as ss
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('cleveland.csv', header=None)
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang','oldpeak', 'slope', 'ca', 'thal', 'target']


df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
df['sex'] = df.sex.map({0: 'female', 1: 'male'})
df['thal'] = df.thal.fillna(df.thal.mean())
df['ca'] = df.ca.fillna(df.ca.mean())

sns.set_context("paper", font_scale=2, rc={"font.size": 20, "axes.titlesize": 25, "axes.labelsize": 20}) 

plt.figure(figsize=(20, 8))
ax = sns.countplot(data=df, x='age', hue='target', order=sorted(df['age'].unique()))
plt.title('Variation of Age for each target class')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

sns.catplot(kind='bar', data=df, y='age', x='sex', hue='target', height=6, aspect=1.5)
plt.title('Distribution of age vs sex with the target class')
plt.tight_layout()
plt.show()

df['sex'] = df.sex.map({'female': 0, 'male': 1})

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for SVM = {:.2f}%'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train) * 100))
print('Accuracy for test set for SVM = {:.2f}%'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test) * 100))
