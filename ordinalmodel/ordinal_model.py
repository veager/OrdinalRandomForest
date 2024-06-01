import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier





class OrdinalClassifierBase(BaseEstimator, ClassifierMixin):
    def __init__(self, probability_joint_type=None, model_params_kwargs=None):

        self.probability_joint_type = 'independent' if probability_joint_type is None else probability_joint_type
        if probability_joint_type not in ['independent', 'conditional']:
            raise ValueError('Invalid "probability_joint_type". Choose from "independent" or "conditional"')

        self.model_params_kwargs = model_params_kwargs
    # -------------------------------------------------
    def init_base_model(self):
        model = RandomForestClassifier(**self.model_params_kwargs)
        return model
    # -------------------------------------------------
    def create_ordinal_target(self, y):
        '''
        convert the target vector ('y') to an ordinal target matrix
        (n_samples, ) -> (n_samples, n_classes - 1)
        the i-th column indicates whether the target is less than or equal to i
        for example,
        y = [0, 1, 1, 2] -->
        ordinal_target = [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 1]
        ]

        param
        ---
        y : shape of (n_samples, )
            the target variable by categorical coding, i.e., 0, 1, 2, ..., K-1

        return:
        ---
        ordinal_target : shape of (n_samples, n_classes - 1)
            the ordinal target matrix
        ---
        '''
        ordinal_target = np.zeros((len(y), self.n_classes_ - 1), dtype=int)

        for i in range(self.n_classes_ - 1):
            # the i-th column indicates whether the target is less than or equal to i
            # 0 : Y(y > i)
            # 1 : Y(y <= i),
            ordinal_target[:, i] = (y <= i)

        return ordinal_target
    # -------------------------------------------
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
        self.ordinal_target = self.create_ordinal_target(y)

        # fit the random forest classifiers
        for i, model in self.estimators_.items():
            model.fit(X, self.ordinal_target[:, i])

        # Return the classifier
        return self
    # -------------------------------------------
    def _predict_proba_raw(self, X):
        '''
        the predicted probability directly derived from the binary classifiers

        model_0 : p_0(x) = P(y <= 0)
        model_1 : p_1(x) = P(y <= 1)
        model_k : p_k(x) = P(y <= k)
        model_K-2 : p_K-2(x) = P(y <= K-2)

        param
        ---
        X:

        return
        ---
        prob (n_samples, n_classes-1):
            prob[: i] = P(y <= i)
        '''
        prob = np.zeros((X.shape[0], self.n_classes_ - 1))

        for i, model in self.estimators_.items():
            prob[:, i] = model.predict_proba(X)[:, 1]

        return prob
    # -------------------------------------------
    def _predict_proba_independent_joint(self, X):
        '''
        # joint probability
        P(y = 0)    =      p_0  * ... *      p_k-1  *      p_k  * ... *      p_K-2
        P(y = k)    = (1 - p_0) * ... * (1 - p_k-1) *      p_k  * ... *      p_K-2
        P(y = K-2)  = (1 - p_0) * ... * (1 - p_k-1) * (1 - p_k) * ... *      p_K-2
        P(y = K-1)  = (1 - p_0) * ... * (1 - p_k-1) * (1 - p_k) * ... * (1 - p_K-2)
        '''
        # (n_samples, n_classes-1)
        prob_raw = self._predict_proba_raw(X)

        # (n_samples, n_classes)
        prob = np.ones((X.shape[0], self.n_classes_))

        for i in range(prob.shape[1]): # prob.shape[1] = n_classes
            # prob[:, i] = 1.
            for j in range(prob_raw.shape[1]): # prob_raw.shape[1] = n_classes - 1
                if j < i:
                    prob[:, i] = prob[:, i] * (1 - prob_raw[:, j])
                else: # j >= i
                    prob[:, i] = prob[:, i] * prob_raw[:, j]

        return prob
    # -------------------------------------------
    def _predict_proba_conditional_joint(self, X):
        '''
        P(y = 0)   = p_0
        P(y = 1)   = p_1   * (1 - p_0)
        P(y = k)   = p_k   * (1 - p_k-1)
        p(y = K-2) = p_K-2 * (1 - p_K-3)
        P(y = K-1) = 1 - P(y <= K-2)
        '''
        # (n_samples, n_classes-1)
        prob_raw = self._predict_proba_raw(X)

        # (n_samples, n_classes)
        prob = np.ones((X.shape[0], self.n_classes_))

        for i, model in self.estimators_.items():
            if i == 0:
                # Y = 0 : P(y = 0) = P(y <= 0)
                prob[:, 0] = prob_raw[:, 0]
            else:
                prob[:, i] = prob_raw[:, i] * (1 - prob_raw[:, i-1])

        prob[:, -1] = 1 - prob_raw[:, -1]

        return prob
    # -------------------------------------------
    def predict_proba(self, X):
        '''
        predict the probability of each class for each sample in X

        :param X:
        :return:
        '''
        check_is_fitted(self)
        X = check_array(X)

        prob = np.zeros((X.shape[0], self.n_classes_))

        self.prob_raw = self._predict_proba_raw(X)

        if self.probability_joint_type == 'independent':
            prob = self._predict_proba_independent_joint(X)
        elif self.probability_joint_type == 'conditional':
            prob = self._predict_proba_conditional_joint(X)

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
        return np.argmax(self.predict_proba(X), axis=1)
    # -------------------------------------------
# ============================================================================



class OrdinalRandomForestClassifier(OrdinalClassifierBase):
    def __init__(self, probability_joint_type, model_params_kwargs):
        super().__init__(
            probability_joint_type = probability_joint_type,
            model_params_kwargs = model_params_kwargs)
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
