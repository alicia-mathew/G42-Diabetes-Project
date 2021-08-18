import pandas as pd
import numpy as np

# Create Multiple ML Transformation Pipelines and test on RFC and XGBoost Classifier algorithms
from implementation_final import baseline_cont_pipeline, baseline_cat_pipeline,\
    cont_pipeline1, cat_pipeline1, cont_pipeline2, cat_pipeline2,\
        cont_pipeline3, cat_pipeline3
from sklearn.compose import ColumnTransformer 
from _01_without_lab_data_driven_prep_for_ML import X, y, cont_cols, cat_cols

baseline_pipeline = ColumnTransformer([
    ('continuous', baseline_cont_pipeline, cont_cols),
    ('categorical', baseline_cat_pipeline, cat_cols),
])

pipeline1 = ColumnTransformer([
    ('continuous', cont_pipeline1, cont_cols),
    ('categorical', cat_pipeline1, cat_cols),
])

pipeline2 = ColumnTransformer([
    ('continuous', cont_pipeline2, cont_cols),
    ('categorical', cat_pipeline2, cat_cols),
])

pipeline3 = ColumnTransformer([
    ('continuous', cont_pipeline3, cont_cols),
    ('categorical', cat_pipeline3, cat_cols),
])

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score

rf_clf = RandomForestClassifier()
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='auc')
algorithms = [rf_clf, xgb_clf]

# find out which RFC transformation pipeline leads to the highest ROC_AUC score
rfc_max_roc_auc = 0
best_rfc_transformation_pipeline = baseline_pipeline # by default before testing others

# find out which XGBoost transformation pipeline leads to the highest ROC_AUC score
xgb_max_roc_auc = 0
best_xgb_transformation_pipeline = baseline_pipeline # by default before testing others

for algorithm in algorithms:
    if algorithm == rf_clf:
        for pipeline in [baseline_pipeline, pipeline1, pipeline2, pipeline3]:
            if pipeline == baseline_pipeline:
                print('############## RFC BASELINE PIPELINE ##########')
            elif pipeline == pipeline1:
                print('############## RFC PIPELINE 1 #################')
            elif pipeline == pipeline2:
                print('############## RFC PIPELINE 2 #################')
            else:
                print('############## RFC PIPELINE 3 #################')

            # Transform X using pipeline
            print('Fitting and transforming X...')
            X_transformed = pipeline.fit_transform(X)
            print('Fitting and transforming X complete.')

            # Fit RFC to transformed X and y 
            rf_clf.fit(X_transformed, y)

            # Evaluate using 5-fold Cross Validation
            print('Performing cross validation...\n')
            cross_val = cross_val_score(rf_clf, X_transformed, y, cv=5)
            mean_cross_val_score = np.mean(cross_val)
            print('Mean cross validation score: {}'.format(mean_cross_val_score))

            y_pred_cv = cross_val_predict(rf_clf, X_transformed, y, cv=5, method='predict_proba')
            roc_auc = roc_auc_score(y, y_pred_cv, multi_class='ovr')
            print('ROC AUC score: {}\n'.format(roc_auc)) 

            if roc_auc > rfc_max_roc_auc:
                rfc_max_roc_auc = roc_auc
                best_rfc_transformation_pipeline = pipeline 

        if best_rfc_transformation_pipeline == baseline_pipeline:
            print('Best RFC transformation pipeline: BASELINE PIPELINE\n')
        elif best_rfc_transformation_pipeline == pipeline1:
            print('Best RFC transformation pipeline: RFC PIPELINE 1\n')
        elif best_rfc_transformation_pipeline == pipeline2:
            print('Best RFC transformation pipeline: RFC PIPELINE 2\n')
        else:
            print('Best RFC transformation pipeline: RFC PIPELINE 3\n')
    else:
        for pipeline in [baseline_pipeline, pipeline1, pipeline2, pipeline3]:
            if pipeline == baseline_pipeline:
                print('############## XGB BASELINE PIPELINE ##########')
            elif pipeline == pipeline1:
                print('############## XGB PIPELINE 1 #################')
            elif pipeline == pipeline2:
                print('############## XGB PIPELINE 2 #################')
            else:
                print('############## XGB PIPELINE 3 #################')

            # Transform X using pipeline
            print('Fitting and transforming X...')
            X_transformed = pipeline.fit_transform(X)
            print('Fitting and transforming X complete.')

            # Fit XGBClassifier to transformed X and y 
            xgb_clf.fit(X_transformed, y)

            # Evaluate using 5-fold Cross Validation
            print('Performing cross validation...\n')
            cross_val = cross_val_score(xgb_clf, X_transformed, y, cv=5)
            mean_cross_val_score = np.mean(cross_val)
            print('Mean cross validation score: {}'.format(mean_cross_val_score))

            y_pred_cv = cross_val_predict(xgb_clf, X_transformed, y, cv=5, method='predict_proba')
            roc_auc = roc_auc_score(y, y_pred_cv, multi_class='ovr')
            print('ROC AUC score: {}\n'.format(roc_auc)) 

            if roc_auc > xgb_max_roc_auc:
                xgb_max_roc_auc = roc_auc
                best_xgb_transformation_pipeline = pipeline 

        if best_xgb_transformation_pipeline == baseline_pipeline:
            print('Best XGB transformation pipeline: BASELINE PIPELINE\n')
        elif best_xgb_transformation_pipeline == pipeline1:
            print('Best XGB transformation pipeline: XGB PIPELINE 1\n')
        elif best_xgb_transformation_pipeline == pipeline2:
            print('Best XGB transformation pipeline: XGB PIPELINE 2\n')
        else:
            print('Best XGB transformation pipeline: XGB PIPELINE 3\n')
