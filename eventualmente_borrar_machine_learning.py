import plotly.express as px

param_distribution = {
    "n_estimators": [10,20,50,70,100],
    "max_features": ["sqrt","log2",None],
    "max_depth": [None, 10,20,30],
    "min_samples_split" : [2,4,8,10],
    "min_samples_leaf" : [1,2,4],
    "bootstrap" : [True, False]
}

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


# Iniciamos el clasificador vacio

rfc = RandomForestClassifier(random_state = 42)
search = RandomizedSearchCV(rfc, param_distribution, n_iter = 10, cv = 5, random_state = 42)
search.fit(X_train,y_train)


search.best_estimator_


search.fit(X_train, y_train)

best_model = search.best_estimator_
best_model

best_params = search.best_params_
best_params

from sklearn.metrics import make_scorer, accuracy_score, classification_report 
print(classification_report(y_train,search.predict(X_train)))

print(classification_report(y_train,dt.predict(X_train)))

## Binchag aplicado a un problema de mas de una clase.

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

# Assuming 'X' is your feature matrix and 'y' is your target vector
# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Split into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Learn to predict each class against the other using OneVsRestClassifier
classifier = OneVsRestClassifier(LinearSVC(random_state=0))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot the ROC curve for each class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    
    
import rasterio as ras

from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.mask import mask
import numpy as np
from shapely.geometry import mapping

show(dataset)

# Leemos los datos como un numpy array
image = dataset.read()

# Flatening
reshaped_image = image.reshape(image.shape[0], -1).T

reshaped_image_no_nan = reshaped_image[~np.isnan(reshaped_image).any(axis=1)]

from sklearn.impute import SimpleImputer
import numpy as np

#predictions = search.predict(reshaped_image_no_nan)


#### Aqui hay una pista de como se hace

https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
https://stackoverflow.com/questions/43331510/how-to-train-an-svm-classifier-on-a-satellite-image-using-python
https://medium.com/@northamericangeoscientistsorg/using-python-to-classify-land-cover-from-satellite-imagery-with-convolutional-neural-networks-328fa3ab0180

## Aqui esta el tutorial definitivo. A disfrutar.

https://gist.github.com/om-henners/c6c8d40389dab75cf535 # clasificacion no supervisada
https://towardsdatascience.com/land-cover-classification-in-satellite-imagery-using-python-ae39dbf2929 # No se puede leer
https://www.youtube.com/watch?v=NFoZPyQqVRA # Clasificacion google earth engine
https://medium.com/@northamericangeoscientistsorg/ # Deep learning deep-learning-for-satellite-image-classification-with-python-ceff1cdf41fb

http://patrickgray.me/open-geo-tutorial/chapter_5_classification.html # Usa la imagen como input para los puntos de entrenamientos
https://geemap.org/notebooks/46_local_rf_training/#save-trees-locally ## Locally trained model with GEE
https://www.sciencedirect.com/science/article/pii/S0303243421001847 # Clasificacion usando deep learning
https://www.linkedin.com/pulse/decision-tree-satellite-image-classification-jo%C3%A3o-otavio/  ## not very useful tutorial

https://ceholden.github.io/open-geo-tutorial/python/chapter_5_classification.html # Interesting

https://www.youtube.com/watch?v=CXlGhiJWKGg

https://github.com/iamtekson/geospatial-machine-learning/blob/main/5.%20Random%20forest%20classification.ipynb