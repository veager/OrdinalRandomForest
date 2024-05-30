import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier





class OrdinalClassifierBase(BaseEstimator, ClassifierMixin):
    def __init__(self, model_params_kwargs):
        self.model_params_kwargs = model_params_kwargs
    # -------------------------------------------------
    def init_base_model(self):
        model = RandomForestClassifier(**self.model_params_kwargs)
        return model
    # -------------------------------------------------
    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # an ordered array of unique labels.
        self.classes_ = unique_labels(y)
        # number of target labels
        self.n_classes_ = len(self.classes_)

        # initialize the random forest classifiers
        self.estimators_ = {i : self.init_base_model() for i in range(self.n_classes_ - 1)}

        # create the ordinal target
        ordinal_target = self.create_ordinal_target(y)

        # fit the random forest classifiers
        for i, model in self.estimators_.items():
            model.fit(X, ordinal_target[:, i])

        # Return the classifier
        return self
    # -------------------------------------------
    def create_ordinal_target(self, y):

        # convert the target vector ('y') to an ordinal target matrix
        # (n_samples, ) -> (n_samples, n_classes - 1)
        # the i-th column indicates whether the target is less than or equal to i
        ordinal_target = np.zeros((len(y), self.n_classes_ - 1), dtype=int)

        for i in range(self.n_classes_ - 1):
            # the i-th column indicates whether the target is less than or equal to i
            # Y(y <= i) = 1, Y(y > i) = 0
            ordinal_target[:, i] = (y <= i)

        return ordinal_target
    # -------------------------------------------
    def predict_prob(self, X):
        '''
        predict the probability of each class for each sample in X
        P(y = 0) = P(y <= 0)
        P(y = 1) = P(y <= 1) - P(y <= 0)
        P(y = k) = P(y <= k) - P(y <= k-1)
        P(y = K - 1) = 1 - P(y <= K-2)
        :param X:
        :return:
        '''
        check_is_fitted(self)
        X = check_array(X)

        prob = np.zeros((X.shape[0], self.n_classes_))
        for i, model in self.estimators_.items():
            if i == 0:
                # Y = 0 : P(y = 0) = P(y <= 0)
                prob[:, i] = model.predict_proba(X)[:, 1]
            else:
                # Y = k : P(y = k) = (1 - P(y <= k-1)) * P(y <= k)
                prob[:, i] = (self.estimators_[i-1].predict_proba(X)[:, 0]) * model.predict_proba(X)[:, 1]
        # Y = K - 1 : P(y = K-1) = 1 - P(y <= K-2)
        prob[:, self.n_classes_-1] = self.estimators_[self.n_classes_-2].predict_proba(X)[:, 0]

        # Normalized
        prob = prob / prob.sum(axis=1)[:, np.newaxis]

        return prob
    # -------------------------------------------
    def predict(self, X):
        '''
        predict the class for each sample in X
        :param X:
        :return:
        '''
        check_is_fitted(self)
        return np.argmax(self.predict_prob(X), axis=1)
    # -------------------------------------------
# ============================================================================



class OrdinalRandomForestClassifier(OrdinalClassifierBase):
    def __init__(self, model_params_kwargs):
        super().__init__(model_params_kwargs)
    # ------------------------------------------------
    def init_base_model(self):
        model = RandomForestClassifier(**self.model_params_kwargs)
        return model
    # -------------------------------------------------
    @property
    def feature_importances_(self):
        check_is_fitted(self)

        feat_imp = []
        for ix, rf in self.estimators_.items():
            # the raw feature importance from each tree
            feat_imp_rf = np.hstack([tree.feature_importances_[:, np.newaxis] for tree in rf.estimators_]) \
                .sum(axis=1)
            # check the consistency of feature importance derived from the two different ways
            # print(feat_imp_rf / feat_imp_rf.sum() - rf.feature_importances_)
            feat_imp.append(feat_imp_rf)

        feat_imp = np.hstack([v[:, np.newaxis] for v in feat_imp])
        # normalizing feature importance
        feat_imp = feat_imp / feat_imp.sum()

        return feat_imp
# ============================================================================
#%%
