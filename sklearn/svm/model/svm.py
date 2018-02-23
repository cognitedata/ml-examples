from sklearn import datasets
from sklearn import svm
from sklearn.externals import joblib

# Load the digits dataset
iris = datasets.load_iris()

# Train a classifier
classifier = svm.SVC()
classifier.fit(iris.data, iris.target)

# Export the classifier to a file
joblib.dump(classifier, 'model.joblib')

# Equivalently, you can use the “pickle” library to export the model similar to:
# import pickle
# with open('model.pkl', 'wb') as model_file:
#   pickle.dump(classifier, model_file)


# The exact file name of of the exported model you upload to GCS is important!
# Your model must be named  “model.joblib”, “model.pkl”, or “model.bst” with respect to
# the library you used to export it.

