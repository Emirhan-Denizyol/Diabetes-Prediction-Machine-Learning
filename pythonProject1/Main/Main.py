from Helpers import config as cng, helpers as hps


cng.pd.set_option("display.max_columns", None)
cng.pd.set_option("display.max_rows", None)
cng.pd.set_option("display.width", 500)
cng.pd.set_option("display.float_format", lambda x: '%.3f' % x)


def main():
    df = cng.pd.read_csv(cng.dataset_path)
    X, y, df = hps.diabetes_data_prep(df)
    hps.base_models(X, y)
    best_models = hps.hyperparameter_optimization(X, y, cv=5)
    voting_clf = hps.voting_classifier(best_models, X, y)
    cng.joblib.dump(voting_clf, "voting_clf2.pkl")

    return voting_clf


if __name__ == "__main__":
    main()
