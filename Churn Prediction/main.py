import pandas as pd

dataset = pd.read_csv('telco.csv')

import numpy as np

print(dataset.isna().sum())
print(dataset.info())
print(dataset.describe())
print(dataset[dataset['TotalCharges'].isna()])

dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')
dataset['TotalCharges'] = dataset['TotalCharges'].astype(float).fillna(dataset['TotalCharges'].median())

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,2)
bar_of_churn = ax[0].bar(['No Churn', 'Churn'],dataset['Churn'].value_counts(), width=0.4)
ax[0].bar_label(bar_of_churn, fmt='%d')
ax[1].pie(dataset['Churn'].value_counts(normalize=True), autopct='%1.1f%%', labels=['No Churn', 'Churn'])
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

dataset.drop('customerID',axis=1, inplace=True)
target = dataset['Churn']
features = dataset.drop('Churn', axis=1)

numerical = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical = ['gender', 'PhoneService', 'MultipleLines','InternetService', 'Contract', 'PaymentMethod', 'Partner','Dependents', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV','StreamingMovies', 'PaperlessBilling']

preprocess = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), categorical)],
    remainder='passthrough'
)

target = target.replace({'No' : 0, 'Yes': 1})

neg_count = target.value_counts().max()
pos_count = target.value_counts().min()

weight = neg_count/pos_count

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import classification_report

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)

models = {
    'random forest' : (RandomForestClassifier(n_jobs=-1, class_weight='balanced'),
                       {'model__n_estimators' : [50, 100, 200], 
                        'model__max_depth' : [3, 5, 10 ,None]}),
    'svm_linear' : (SVC(max_iter=5000, class_weight='balanced'), 
             {'model__C' : [0.1, 1, 10, 100],
              'model__kernel' : ['linear']}),
    'svm_others' : (SVC(max_iter=5000, class_weight='balanced'),
                    {'model__C' : [0.1, 1, 10, 100],
                     'model__kernel' : ['rbf', 'poly'],
                     'model__gamma' : ['scale', 'auto']}),
    'xgboost' : (XGBClassifier(random_state=42, n_jobs=-1, scale_pos_weight=weight),
                              {'model__n_estimators' : [50, 100, 200],
                               'model__max_depth' : [3, 5, 10]}),
    'logreg' : (LogisticRegression(random_state=42, n_jobs=-1, class_weight='balanced'),
                {'model__C' : [0.01, 0.1, 1 ,10],
                 'model__solver' : ['lbfgs', 'liblinear']})
}


def best_models(models):
    
    saved_params = {}
    
    for name, model in models.items():
        
        pipeline = Pipeline([
            ('preprocess', clone(preprocess)),
            ('model', model[0])
        ])
        
        grid = GridSearchCV(pipeline, model[1], scoring=['recall','precision', 'f1', 'average_precision'], cv=5, refit='recall', n_jobs=-1)
        grid.fit(x_train, y_train)
        if name == 'xgboost':
            pure_params = {k.split('model__')[1]: v for k, v in grid.best_params_.items() if k.startswith('model__')}
            saved_params[name] = pure_params
        results = pd.DataFrame(grid.cv_results_)
        res = results.iloc[grid.best_index_]
        
        print(f"{name}")
        print(f"recall:    {res['mean_test_recall']:.3f}")
        print(f"precision: {res['mean_test_precision']:.3f}")
        print(f"f1:        {res['mean_test_f1']:.3f}")
        print(f"pr-auc:   {res['mean_test_average_precision']:.3f}\n")
        print(f"test:   \n{classification_report(y_test, grid.best_estimator_.predict(x_test))}")
    
    return saved_params
        
        
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

params_dict = best_models(models)

final_model = XGBClassifier(**params_dict['xgboost'])

pipeline = Pipeline([
    ('preprocess', preprocess),
    ('model', final_model)
])
pipeline.fit(x_train, y_train)

matrix = ConfusionMatrixDisplay.from_estimator(pipeline, x_test, y_test, cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.show()

feature_columns = pipeline.named_steps['preprocess'].get_feature_names_out()
importance = pd.Series(final_model.feature_importances_, index=feature_columns).sort_values(ascending=False).head(10).round(2)

print(f'Importance\n{importance}')

y_proba = pipeline  .predict_proba(x_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label= 'ROC Curve')
plt.plot([0,1], [0,1], linestyle= '--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()