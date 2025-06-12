# %%
import pandas
import numpy
from sklearn import *
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import pickle

# %% [markdown]
# Load dataset

# %%
DATA_FILE_URL = 'https://raw.githubusercontent.com/SasidharSekar/Classification-abalone-sex/refs/heads/main/abalone-data.csv'
col_names = ['Sex','Length','Diameter','Height','Whole Weight','Shucked Weight','Viscera Weight','Shell Weight','Rings']
data = pandas.read_csv(DATA_FILE_URL,sep=',',quotechar='"', header=None, names=col_names)

# %% [markdown]
# View Data Distribution

# %%
print("Data Size: %d" %data.size)
print(data.head(10))
print(data.describe())
print(data.groupby('Sex').size())
excl_gender = data.iloc[:,1:]
print(excl_gender.corr())

# %% [markdown]
# Visualize Data Distribution

# %%
data.hist()
pyplot.show()
X = data.iloc[:,1:]
X.boxplot()
pyplot.show()
scatter_matrix(data)
pyplot.show()

# %% [markdown]
# Model Evaluation Preparation

# %%
array = data.values
X = array[:,1:]
y = array[:,0]
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=1)
scaler = MinMaxScaler()
scaler.fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_val = scaler.transform(X_val)

models = []
models.append(("LR",LogisticRegression()))
models.append(("LDA",LinearDiscriminantAnalysis()))
models.append(("KNN",KNeighborsClassifier()))
models.append(("CART",DecisionTreeClassifier()))
models.append(("NB",GaussianNB()))
models.append(("SVM",SVC(gamma="auto")))

# %% [markdown]
# Model Evaluation

# %%
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10,shuffle=True, random_state=1)
    cv_results = cross_val_score(model,scaled_X_train,y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f(%f)' %(name, cv_results.mean(), cv_results.std()))

# %% [markdown]
# Compare Algorithms

# %%
pyplot.boxplot(results,tick_labels=names)
pyplot.title('Algorithm Comparision') 
pyplot.show()

# %% [markdown]
# Make Predictions

# %%
model = LogisticRegression()
model.fit(scaled_X_train,y_train)
predictions = model.predict(scaled_X_val)
print(accuracy_score(y_val,predictions))
print(confusion_matrix(y_val,predictions))
print(classification_report(y_val,predictions))

# %% [markdown]
# Make individual predictions

# %%
str_x_test = input("Enter input parameters as Comma Separated values")
x_test = str_x_test.split(",")
X_test = numpy.array(x_test)
X_test = X_test.reshape(1,-1)
X_test = X_test.astype(float)
scaled_X_test = scaler.transform(X_test)
prediction = model.predict(scaled_X_test)
print(prediction)

# %% [markdown]
# Save Model to File

# %%
filename = 'final_model_classification_abalone_sex.sav'
pickle.dump(model, open(filename,'wb'))

# %% [markdown]
# Load Model from File and predict

# %%
model = pickle.load(open(filename,'rb'))
predictions = model.predict(scaled_X_val)
print(accuracy_score(y_val,predictions))


