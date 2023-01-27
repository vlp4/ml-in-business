import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score, log_loss

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]


class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]


class OHEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
        self.columns = None

    def fit(self, X, y=None):
        if not self.columns:
            self.columns = [col for col in pd.get_dummies(X, prefix=self.key).columns]
            if self.key == 'workclass':
                x = 1
                pass
        return self

    def transform(self, X):
        X = pd.get_dummies(X, prefix=self.key)
        for col_ in self.columns:
            if col_ not in X.columns:
                X[col_] = 0
        return X[self.columns]


def split_y(df, y_column):
    return df.drop(columns=y_column), df[y_column]


def measure(pipe, X_test, y_test, title):
    y_score = pipe.predict_proba(X_test)[:, 1]

    b = 1
    precision, recall, thresholds = precision_recall_curve(y_test.values, y_score)
    fscore = (1 + b ** 2) * (precision * recall) / (b ** 2 * precision + recall)
    # locate the index of the largest f score
    ix = np.nanargmax(fscore)
    roc_auc = roc_auc_score(y_true=y_test, y_score=pipe.predict_proba(X_test)[:, 1])
    logloss = log_loss(y_true=y_test, y_pred=pipe.predict_proba(X_test)[:, 1])
    print(f'{title}:\n  Best-Threshold={thresholds[ix]:4.3f}  F-Score={fscore[ix]:4.3f}  Precision={precision[ix]:4.3f}  '
          f'Recall={recall[ix]:4.3f}  roc_auc={roc_auc:4.3f}  log_loss={logloss:4.3f}')


def make_model():
    df = pd.read_csv('train_data/adult.data', names=[
        'age',
        'workclass',
        'fnlwgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'native-country',
        'rich'
    ])
    print(df.head())

    df = df.applymap(lambda v: v.strip() if isinstance(v, str) else v)
    df['rich'] = df['rich'].map(lambda val: 1 if val.strip() == '>50K' else 0)

    continuos_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    base_cols = []

    # Build the pipeline
    continuos_transformers = []
    cat_transformers = []
    base_transformers = []
    for cont_col in continuos_cols:
        transfomer = Pipeline([
            ('selector', NumberSelector(key=cont_col)),
            ('standard', StandardScaler())
        ])
        continuos_transformers.append((cont_col, transfomer))

    for cat_col in cat_cols:
        cat_transformer = Pipeline([
            ('selector', ColumnSelector(key=cat_col)),
            ('ohe', OHEEncoder(key=cat_col))
        ])
        cat_transformers.append((cat_col, cat_transformer))

    for base_col in base_cols:
        base_transformer = Pipeline([
            ('selector', NumberSelector(key=base_col))
        ])
        base_transformers.append((base_col, base_transformer))

    feats = FeatureUnion(continuos_transformers + cat_transformers + base_transformers)
    feature_pipe = Pipeline([('feats', feats)])
    feature_pipe.fit(df)

    pipe = Pipeline([
        ('features', feature_pipe),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=200, max_depth=4))
        # ('classifier', LogisticRegression(random_state=42))
        # ('classifier', SGDClassifier(random_state=42, loss='log_loss'))
        # ('classifier', GradientBoostingClassifier(random_state=42),)
    ])

    target_column = 'rich'

    df_train, df_test = train_test_split(df, train_size=0.5, random_state=42)
    X_train, y_train = split_y(df_train, target_column)
    X_test, y_test = split_y(df_test, target_column)

    # Обучаем классификатор
    pipe.fit(X_train, y_train)
    measure(pipe, X_test, y_test, 'Результат обучения на обучающей выборке')
    return pipe
