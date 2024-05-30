import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics


def print_performance(y_true, y_pred, digits=3):

    print(
        '\nAccuracy:',
        np.round(sklearn.metrics.accuracy_score(y_true, y_pred), digits),
        '\nConfusion matrix:\n',
        np.round(sklearn.metrics.confusion_matrix(y_true, y_pred), digits),
        '\nClassification report: \n',
        sklearn.metrics.classification_report(y_true, y_pred, digits=digits),
    )

    return None
# =============================================================================
def convert_numerical_to_categorical(y, mapper=None):
    '''
    Convert numerical target variable into categorical target variable

    Parameters
    ----------
    y : pd.Series
        The target variable.
    mapper : dict, optional
        a mapper mapping numerical values to categorical values
    '''
    y = y.map(mapper)

    category_ordered = []
    for k, v in dict(sorted(mapper.items(), key=lambda x: x[0])).items():
        if not (v in category_ordered):
            category_ordered.append(v)
    print(category_ordered)

    y = pd.Series(pd.Categorical(y, categories=category_ordered, ordered=True), index=y.index, name=y.name)

    return y
# =============================================================================
def convert_categorical_to_numerical(y):
    '''
    Convert categorical target variable into numerical target variable

    Parameters
    ----------
    y : pd.Series
        The target variable.
    '''
    y = pd.Series(y.cat.codes, index=y.index, name=y.name)
    return y
# =============================================================================
def convert_target(y, object_type, mapper=None):
    '''
    Parameters
    ----------
    y : pd.Series
        The target variable.
    object_type : str in {'category', 'int'}
        The type of target variable.
    mapper : dict, optional
        The mapping between numerical and categorical variables
    '''

    from pandas.api.types import is_integer_dtype

    if not (is_integer_dtype(y) or isinstance(y.dtype, pd.CategoricalDtype)):
        raise ValueError('The target variable must be either "int" or "category"')

    # Convert numerical into categorical
    if is_integer_dtype(y) and (object_type == 'category'):
        if mapper is None:
            raise ValueError('The argument "mapper" must be provided')

        y = convert_numerical_to_categorical(y, mapper=mapper)

    # Convert categorical into numerical
    elif isinstance(y.dtype, pd.CategoricalDtype) and (object_type == 'int'):
        y = convert_categorical_to_numerical(y)

    else:
        raise ValueError('The object data type is same to the input data type')

    return y
# =============================================================================