import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
st.set_option('deprecation.showPyplotGlobalUse', False)

#TITLE
st.title('CLASSIFICATION DEMO')
image = Image.open('mother_nature.jpg')
st.image(image,use_column_width=True)

#SUBTITLE
st.write("Iris and Wine Classification with ML")
#SELECT DATASET
df_name=st.sidebar.selectbox("Select dataset",('Iris','Wine'))
st.write(f"## {df_name} Dataset")

#SELECT CLASSIFIER
model = st.sidebar.selectbox('Select model',('KNN','SVM'))
st.write(f"## {model} Model")

#FUNCTION TO LOAD DATSET
def get_datset(name):
	data = None
	if name =='Iris':
		data=datasets.load_iris()
	# elif name=='Wine':
	# 	data=datasets.load_wine()
	else:
		#data = datasets.load_breast_cancer()
		data=datasets.load_wine()
	X=data.data
	y=data.target
	return X,y
X,y = get_datset(df_name)
st.dataframe(X)
st.write("Shape",X.shape)
st.write('Target shape',y.shape)
st.write("Number of classes", len(np.unique(y)))

# fig = plt.figure()
# sns.boxplot(data=X, orient='h')

# st.pyplot()
# #PLOT HISTOGRAM
# plt.hist(X)
# st.pyplot()
#BUILD THE MODELS PARAMETERS
def add_params(name_of_clf):
	params = dict()
	if name_of_clf=="SVM":
		C=st.sidebar.slider('C', 0.01, 15.0)
		params['C']= C
	else:
		name_of_clf=="KNN"
		K=st.sidebar.slider('K',1,15)
		params['K']=K
	return params
params=add_params(model)

#ACCESSING OUR CLASSIFIER
def get_classifier(name_of_clf,params):
	clf=None
	if name_of_clf=="SVM":
		clf=SVC(C=params['C'])
	elif name_of_clf == "KNN":
		clf=KNeighborsClassifier(n_neighbors=params['K'])
	else:
		st.warnings('Select a model')
	return clf
clf = get_classifier(model,params)

#BUILD THE MODEL
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=.3,random_state=1)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {model}')
st.write(f'Accuracy =', accuracy)


#PLOT DATASET
pca = PCA(2)
X_projected = pca.fit_transform(X)
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')

plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.colorbar()

plt.show()
st.pyplot()
