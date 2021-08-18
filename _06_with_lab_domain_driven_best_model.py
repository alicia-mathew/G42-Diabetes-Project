import pandas as pd
import numpy as np

# Use best RFC transformation pipeline and best XGB transformation pipeline to perform Grid Search
from _05_with_lab_domain_driven_test_pipelines import X, y, cont_cols_dom, cat_cols_dom, \
    best_rfc_transformation_pipeline, best_xgb_transformation_pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import roc_auc_score

rf_clf = RandomForestClassifier()
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='auc')
best_rfc_estimator_roc_auc = 0 
best_xgb_estimator_roc_auc = 0
algorithms = [rf_clf, xgb_clf]

for algorithm in algorithms:
    if algorithm == rf_clf:
        # Transform X using best RFC transformation pipeline
        print('############## RANDOM FOREST CLASSIFIER #######\n')
        print('############## BEST RFC PIPELINE ##############')
        print('Fitting and transforming X...')
        X_transformed = best_rfc_transformation_pipeline.fit_transform(X)
        print('Fitting and transforming X complete.\n')

        # Perform GridSearchCV and Train X
        param_grid = {'n_estimators': [30, 50, 70, 100], 'max_features': ['auto'],\
            'min_samples_leaf': [1, 3, 5], 'min_samples_split': [2, 3, 4]}

        print('Performing Grid Search...')
        print('Parameter grid: {}\n'.format(param_grid))
        grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=5, verbose=2,\
            scoring='accuracy')
        grid_search.fit(X_transformed, y)

        # Store the best RFC estimator
        best_rfc_estimator = grid_search.best_estimator_
        print('Best RFC estimator: {}\n'.format(best_rfc_estimator))

        # Evaluation metrics
        from sklearn.metrics import roc_auc_score
        mean_cross_val_score = grid_search.best_score_
        print('Mean cross validation score: {}'.format(mean_cross_val_score))

        y_pred_cv = cross_val_predict(best_rfc_estimator, X_transformed, y, cv=5, method="predict_proba")
        best_rfc_estimator_roc_auc = roc_auc_score(y, y_pred_cv, multi_class='ovr')
        print('ROC AUC score: {}\n'.format(best_rfc_estimator_roc_auc))
    else:
        # Transform X using best XGB transformation pipeline
        print('############## XGBOOST CLASSIFIER #############\n')
        print('############## BEST XGB PIPELINE ##############')
        print('Fitting and transforming X...')
        X_transformed = best_xgb_transformation_pipeline.fit_transform(X)
        print('Fitting and transforming X complete.\n')

        # Perform GridSearchCV and Train X
        param_grid = {'learning_rate': [0.1, 0.2, 0.3], 'max_depth': [4, 5, 6], 'subsample': [0.5, 0.7, 1],\
            'gamma': [0, 0.1, 0.5], 'reg_alpha': [0.005, 0.01, 0.05]} 
        # where to split, values of n 

        xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='auc')

        print('Performing Grid Search...')
        print('Parameter grid: {}\n'.format(param_grid))
        grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=5, verbose=2,\
            scoring='accuracy')
        grid_search.fit(X_transformed, y)

        # Store the best XGB estimator
        best_xgb_estimator = grid_search.best_estimator_
        print('Best XGB estimator: {}\n'.format(best_xgb_estimator))

        # Evaluation metrics
        from sklearn.metrics import roc_auc_score
        mean_cross_val_score = grid_search.best_score_
        print('Mean cross validation score: {}'.format(mean_cross_val_score))

        y_pred_cv = cross_val_predict(best_xgb_estimator, X_transformed, y, cv=5, method="predict_proba")
        best_xgb_estimator_roc_auc = roc_auc_score(y, y_pred_cv, multi_class='ovr')
        print('ROC AUC score: {}\n'.format(best_xgb_estimator_roc_auc))

print('############## BEST MODEL #####################\n')
if best_xgb_estimator_roc_auc > best_rfc_estimator_roc_auc:
    best_model = best_xgb_estimator
    print('#### BEST MODEL: XGB CLF AFTER GRID SEARCH ####\n')
    print('ROC AUC score: {}\n'.format(best_xgb_estimator_roc_auc))
else:
    best_model = best_rfc_estimator
    print('###### BEST MODEL: RFC AFTER GRID SEARCH ######\n')
    print('ROC AUC score: {}\n'.format(best_rfc_estimator_roc_auc))

print('############## BEST TRANSFORMATION PIPELINE ###\n')
if best_model == best_xgb_estimator:
    best_transformation_pipeline = best_xgb_transformation_pipeline
else:
    best_transformation_pipeline = best_rfc_transformation_pipeline

# Feature Importance
one_hot_columns = best_transformation_pipeline.named_transformers_['categorical'].\
    named_steps['one_hot_encoder'].get_feature_names(input_features=cat_cols_dom)
all_columns = cont_cols_dom + list(one_hot_columns)
top_feature_names = list(np.array(all_columns)[best_model.feature_importances_.argsort()[::-1]])
top_feature_weights = list(np.array(best_model.feature_importances_)[best_model.feature_importances_.argsort()[::-1]])
top_features_df = pd.DataFrame({'Feature': top_feature_names, 'Weight': top_feature_weights})
print('############## TOP 20 FEATURES  ###############\n')
print(top_features_df.head(20))

# Display Feature Importance
import matplotlib.pyplot as plt

plt.barh(top_features_df.head(20)['Feature'][::-1], top_features_df.head(20)['Weight'][::-1])
plt.title('Top 20 Features (Domain-Driven Approach)')
plt.xlabel('Feature Importance')
plt.ylabel('Feature Names')
plt.show()
