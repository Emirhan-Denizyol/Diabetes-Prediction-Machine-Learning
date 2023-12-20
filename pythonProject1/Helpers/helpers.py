from Helpers import config as cng

# Data Preprocessing & Future Engineering


def species_identification(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        Numerik fakat kategorik değişkenler için sınıf eşik değeri
    car_th: int, float
        Kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi
    num_but_cat: list
        Numerik görünümlü kategorik değişken listesi

    Notes
    -------
    cat_cols + num_cols + cat_but_car = Toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde
    Return olan 3 liste toplamı toplam değişken sayısına eşittir.
    """
    # cat_cols and cat_but_car
    cat_cols = [col for col in dataframe.columns if
                str(dataframe[col].dtype) in ["category", "bool", "object"]]
    num_but_cat = [col for col in dataframe.columns if
                   str(dataframe[col].dtype) in ["int64", "float64"] and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtype) in ["category", "bool", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations : {dataframe.shape[0]}")
    print(f"Variables : {dataframe.shape[1]}")
    print(f"cat_cols : {len(cat_cols)}")
    print(f"num_cols : {len(num_cols)}")
    print(f"cat_but_car : {len(cat_but_car)}")
    print(f"num_but_car : {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car


def outliers_threshold(dataframe, col_name, q1=0.25, q3=0.75, floor_number=1.5):
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    iqr = quantile3 - quantile1
    upper_limit = quantile3 + floor_number * iqr
    lower_limit = quantile1 - floor_number * iqr
    return lower_limit, upper_limit


def replace_with_threshold(dataframe, variable, q1=0.25, q3=0.75):
    low_limit, up_limit = outliers_threshold(dataframe=dataframe, col_name=variable, q1=q1, q3=q3, floor_number=1.5)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def one_hot_encoding(dataframe, categorical_columns, drop_first=True):
    dataframe = cng.pd.get_dummies(dataframe, columns=categorical_columns, drop_first=drop_first)
    return dataframe


def diabetes_data_prep(dataframe):
    dataframe.columns = [col.upper() for col in dataframe.columns]

    # GLUCOSE
    dataframe['NEW_GLUCOSE_CAT'] = cng.pd.cut(x=dataframe['GLUCOSE'], bins=[-1, 139, 200], labels=['normal', 'prediabetes'])

    # AGE
    dataframe.loc[(dataframe['AGE'] < 35), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 35) & (dataframe['AGE'] <= 55), 'NEW_AGE_CAT'] = 'middleage'
    dataframe.loc[(dataframe['AGE'] > 55), 'NEW_AGE_CAT'] = 'old'

    # BMI
    dataframe['NEW_BMI_RANGE'] = cng.pd.cut(x=dataframe['BMI'], bins=[-1, 18.5, 24.9, 29.9, 100], labels=['underweight', 'healty', 'overweight', 'obese'])

    # BLOODPRESSURE
    dataframe['NEW_BLOODPRESSURE'] = cng.pd.cut(x=dataframe['BLOODPRESSURE'], bins=[-1, 79, 89, 123], labels=['normal', 'hs1', 'hs2'])

    cat_cols, num_cols, cat_but_car = species_identification(dataframe, cat_th=5, car_th=20)

    cat_cols = [col for col in cat_cols if 'OUTCOME' not in col]

    dataframe = one_hot_encoding(dataframe, cat_cols, drop_first=True)

    dataframe.columns = [col.upper() for col in dataframe.columns]

    cat_cols, num_cols, cat_but_car = species_identification(dataframe, cat_th=5, car_th=20)

    cat_cols = [col for col in cat_cols if 'OUTCOME' not in col]

    replace_with_threshold(dataframe, 'INSULIN')

    X_scaled = cng.StandardScaler().fit_transform(dataframe[num_cols])  # Numpy Arraye dönüşüyor

    dataframe[num_cols] = cng.pd.DataFrame(data=X_scaled, columns=dataframe[num_cols].columns)

    y = dataframe['OUTCOME']
    X = dataframe.drop(['OUTCOME'], axis=1)

    return X, y, dataframe


# Base Model

def base_models(X, y, scoring='roc_auc'):
    print('Base Models...........')

    classifiers = [('LR', cng.LogisticRegression()),
                   ('KNN', cng.KNeighborsClassifier()),
                   ('SVC', cng.SVC()),
                   ('CART', cng.DecisionTreeClassifier()),
                   ('RF', cng.RandomForestClassifier()),
                   ('AdaBoost', cng.AdaBoostClassifier()),
                   ('GBM', cng.GradientBoostingClassifier()),
                   ('XGBoost', cng.XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', cng.LGBMClassifier()),
                   ('CATBoost', cng.CatBoostClassifier())]

    for name, classifier in classifiers:
        cv_results = cng.cross_validate(classifier, X, y, cv=5, scoring=scoring)
        print(f"{scoring} : {round(cv_results['test_score'].mean(), 4)} ({name})")


def hyperparameter_optimization(X, y, cv=3, scoring='roc_auc'):
    print("Hyperparameter Optimizataion..........")

    best_models = {}

    for name, classifier, params in cng.classifiers:
        print(f"############{name}############")

        cv_results = cng.cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before) : {round(cv_results['test_score'].mean(), 4)} ({name})")

        gs_best = cng.GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cng.cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After) : {round(cv_results['test_score'].mean(), 4)} ({name})")
        print(f"{name} best params : {gs_best.best_params_}", end="\n\n\n")

        best_models[name] = final_model

    return best_models


# Stacking & Ensemble Learning


def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = cng.VotingClassifier(estimators=[('KNN', best_models['KNN']),
                                                  ('RF', best_models['RF']),
                                                  ('CART', best_models['CART']),
                                                  ('XGBoost', best_models['XGBoost']),
                                                  ('LightGBM', best_models['LightGBM'])],
                                      voting='soft').fit(X, y)

    cv_results = cng.cross_validate(voting_clf,
                                    X,
                                    y,
                                    cv=5,
                                    scoring=['accuracy', 'f1', 'roc_auc'])

    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1: {cv_results['test_f1'].mean()}")
    print(f"Roc AUC: {cv_results['test_roc_auc'].mean()}")

    return voting_clf