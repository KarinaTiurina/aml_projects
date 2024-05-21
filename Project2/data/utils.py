from typing import Optional, List, Tuple, Callable, Dict

import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, BaggingClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Perceptron


def cleanup_dataset_remove_features_vif(
        df: pd.DataFrame,
        target: Optional[str | int] = None,
        threshold: float = 10.0
) -> List[str | int]:
    """
    Remove features with VIF greater than threshold.
    :param df: DataFrame - input data
    :param target: str - target column name
    :param threshold: float - VIF threshold
    :return: List[str | int] - list of removed features
    """

    features = list(df.columns)
    if target is not None:
        features.remove(target)

    all_removed_features = []
    while True:
        vif = pd.DataFrame()
        vif["features"] = features
        vifs = []

        tqdm._instances.clear()
        bar = tqdm(total=len(features), desc="Calculating VIF")
        for i in range(len(features)):
            vifs.append(variance_inflation_factor(df[features].values, i))
            bar.update(1)

        bar.close()
        vif["VIF"] = vifs

        max_vif = vif['VIF'].max()
        if max_vif > threshold:
            print(f"Removing feature {vif.loc[vif['VIF'].idxmax()]['features']} with VIF {max_vif}")
            vif_max_feature = vif.loc[vif['VIF'].idxmax()]['features']
            features.remove(vif_max_feature)
            all_removed_features.append(vif.loc[vif['VIF'].idxmax()]['features'])
        else:
            break

    return all_removed_features


def cleanup_dataset_apply_standard_scaler(
        df: pd.DataFrame,
        target: Optional[str | int] = None
) -> pd.DataFrame:
    """
    Apply StandardScaler to the features

    :param df: DataFrame - input data
    :param target: str - target column name

    :return: DataFrame - cleaned data
    """
    df = df.copy()

    features = list(df.columns)
    if target is not None:
        features.remove(target)
    scaler = StandardScaler()

    if target is not None:
        df[features] = scaler.fit_transform(df[features])

    return df


class MultiCollinearityEliminator:

    def __init__(
            self,
            df: pd.DataFrame,
            threshold: float = 0.9
    ):
        """
        MultiCollinearityEliminator class is created to eliminate multicollinearity from the dataframe.
        From: https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on

        :param df: DataFrame - input data
        :param threshold: float - threshold value to consider the correlation between the features
        """

        self.df = df
        self.threshold = threshold

    def __create_correlation_matrix(self, current_filtered_features: List[str | int]) -> pd.DataFrame:
        """
        Create the correlation matrix of the dataframe.
        If include_target is True, the target column will be included in the correlation matrix.

        :param current_filtered_features: List[str | int] - list of features to be excluded from the correlation matrix
        :return: DataFrame - correlation matrix
        """
        # Checking we should include the target in the correlation matrix
        df_temp = self.df.drop(columns=current_filtered_features)
        # Setting method to Pearson to prevent issues in case the default method for df.corr() gets changed
        # Setting min_period to 30 for the sample size to be statistically significant (normal) according to
        # central limit theorem
        correlation_matrix = df_temp.corr(method='pearson', min_periods=30).abs()

        return correlation_matrix

    def __create_correlated_features_list(
            self,
            current_filtered_features: List[str | int]
    ) -> List[str | int]:
        """
        Create the list of correlated features based on the threshold value.

        :return: list - list of correlated features
        """

        # Obtaining the correlation matrix of the dataframe (without the target)
        correlation_matrix = self.__create_correlation_matrix(current_filtered_features)
        correlated_columns = []

        # Iterating through the columns of the correlation matrix dataframe
        for column in correlation_matrix.columns:
            # Iterating through the values (row wise) of the correlation matrix dataframe
            for idx, row in correlation_matrix.iterrows():
                if (row[column] > self.threshold) and (row[column] < 1):
                    # Adding the features that are not already in the list of correlated features
                    if idx not in correlated_columns:
                        correlated_columns.append(idx)
                    if column not in correlated_columns:
                        correlated_columns.append(column)

        return correlated_columns

    def auto_eliminate_multicollinearity(self) -> List[str | int]:
        """
        Automatically eliminate multicollinearity from the dataframe.

        :return: List[str | int] - list of removed features
        """

        # Obtaining the list of correlated features
        correlated_columns = self.__create_correlated_features_list([])

        all_correlated_columns = correlated_columns.copy()
        while correlated_columns:
            # Obtaining the dataframe after deleting the feature (from the list of correlated features)
            # that is least correlated with the target
            correlated_columns = self.__create_correlated_features_list(all_correlated_columns)
            all_correlated_columns.extend(correlated_columns)

        return all_correlated_columns


def cleanup_dataset_remove_features_correlation(
        df: pd.DataFrame,
        target: Optional[str | int] = None,
        threshold: float = 0.9
) -> List[str | int]:
    """
    Remove features with correlation greater than threshold.

    :param df: DataFrame - input data
    :param target: str - target column name
    :param threshold: float - correlation threshold
    :return: List[str | int] - list of removed features
    """
    df = df.copy()
    if target is not None:
        df = df.drop(columns=[target])

    eliminator = MultiCollinearityEliminator(df, threshold)
    return eliminator.auto_eliminate_multicollinearity()


SelectFeaturesMethod = Callable[[pd.DataFrame, pd.Series, int], pd.Series]


def select_features_rfe_logistic_regression(X: pd.DataFrame, y: pd.Series, random_state: int = 0) -> pd.Series:
    """
    Select features using Recursive Feature Elimination with Logistic Regression

    :param X: DataFrame - input data
    :param y: DataFrame - target data
    :param random_state: int - random state
    :return: Series - selected features
    """
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    rfe = RFE(model, n_features_to_select=100)
    rfe = rfe.fit(X, y)
    feature_mask = rfe.support_
    return X.columns[feature_mask]


def select_features_rfe_random_forest(X: pd.DataFrame, y: pd.Series, random_state: int = 0) -> pd.Series:
    """
    Select features using Recursive Feature Elimination with Random Forest

    :param X: DataFrame - input data
    :param y: DataFrame - target data
    :param random_state: int - random state
    :return: Series - selected features
    """
    model = RandomForestClassifier(n_jobs=-1, random_state=random_state)
    rfe = RFE(model, n_features_to_select=100)
    rfe = rfe.fit(X, y)
    feature_mask = rfe.support_
    return X.columns[feature_mask]


def select_features_rfe_support_vector_machine(X: pd.DataFrame, y: pd.Series, random_state: int = 0) -> pd.Series:
    """
    Select features using Recursive Feature Elimination with Support Vector Machine

    :param X: DataFrame - input data
    :param y: DataFrame - target data
    :param random_state: int - random state
    :return: Series - selected features
    """
    model = LinearSVC(random_state=random_state, dual=False)
    rfe = RFE(model, n_features_to_select=100)

    rfe = rfe.fit(X, y)
    feature_mask = rfe.support_
    return X.columns[feature_mask]


def select_features_random_forest(X: pd.DataFrame, y: pd.Series, random_state: int = 0) -> pd.Series:
    """
    Select features using Random Forest

    :param X: DataFrame - input data
    :param y: DataFrame - target data
    :param random_state: int - random state
    :return: Series - selected features
    """
    model = RandomForestClassifier(n_jobs=-1, random_state=random_state)
    model = model.fit(X, y)
    sfm = SelectFromModel(model, prefit=True, max_features=100)
    feature_mask = sfm.get_support()
    return X.columns[feature_mask]


def select_features_perceptron(X: pd.DataFrame, y: pd.Series, random_state: int = 0) -> pd.Series:
    """
    Select features using Perceptron

    :param X: DataFrame - input data
    :param y: DataFrame - target data
    :param random_state: int - random state
    :return: Series - selected features
    """
    model = Perceptron(n_jobs=-1, random_state=random_state)
    model = model.fit(X, y)
    sfm = SelectFromModel(model, prefit=True, max_features=100)
    feature_mask = sfm.get_support()
    return X.columns[feature_mask]


feature_selectors: Dict[str, SelectFeaturesMethod] = {
    'random_forest': select_features_random_forest,
    'perceptron': select_features_perceptron,
    'rfe_support_vector_machine': select_features_rfe_support_vector_machine,
    'rfe_logistic_regression': select_features_rfe_logistic_regression,
    'rfe_random_forest': select_features_rfe_random_forest,
}

ReduceDimensionalityMethod = Callable[[pd.DataFrame], pd.DataFrame]


def reduce_dimensionality_pca(X: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce dimensionality using PCA

    :param X: DataFrame - input data
    :return: DataFrame - reduced data
    """
    pca = PCA(n_components=100, random_state=0)
    return pd.DataFrame(pca.fit_transform(X))


dimensionality_reducers: Dict[str, ReduceDimensionalityMethod] = {
    'pca': reduce_dimensionality_pca
}

ApplyModelMethod = Callable[[pd.DataFrame, pd.Series, pd.DataFrame, Optional[int]], pd.Series]


def apply_model_logistic_regression(
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame,
        random_state: int = 0
) -> pd.Series:
    """
    Apply Logistic Regression model

    :param X: DataFrame - input data
    :param y: Series - target data
    :param X_test: DataFrame - test data
    :param random_state: int - random state
    :return: Series - predictions on the test data
    """
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model = model.fit(X, y)
    return model.predict(X_test)


def apply_model_random_forest(
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame,
        random_state: int = 0
) -> pd.Series:
    """
    Apply Random Forest model

    :param X: DataFrame - input data
    :param y: Series - target data
    :param X_test: DataFrame - test data
    :param random_state: int - random state
    :return: Series - predictions on the test data
    """
    model = RandomForestClassifier(n_jobs=-1, random_state=random_state)
    model = model.fit(X, y)
    return model.predict(X_test)


def apply_model_gradient_boosting_classifier(
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame,
        random_state: int = 0
) -> pd.Series:
    """
    Apply Gradient Boosting Classifier model

    :param X: DataFrame - input data
    :param y: Series - target data
    :param X_test: DataFrame - test data
    :param random_state: int - random state
    :return: Series - predictions on the test data
    """
    model = HistGradientBoostingClassifier(random_state=random_state)
    model = model.fit(X, y)
    return model.predict(X_test)


def apply_model_support_vector_machine(
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame,
        random_state: int = 0
) -> pd.Series:
    """
    Apply Support Vector Machine model

    :param X: DataFrame - input data
    :param y: Series - target data
    :param X_test: DataFrame - test data
    :param random_state: int - random state
    :return: Series - predictions on the test data
    """
    model = LinearSVC(random_state=random_state, dual=False)
    model = model.fit(X, y)
    return model.predict(X_test)


def apply_model_perceptron(
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame,
        random_state: int = 0
) -> pd.Series:
    """
    Apply Perceptron model

    :param X: DataFrame - input data
    :param y: Series - target data
    :param X_test: DataFrame - test data
    :param random_state: int - random state
    :return: Series - predictions on the test data
    """
    model = Perceptron(n_jobs=-1, random_state=random_state)
    model = model.fit(X, y)
    return model.predict(X_test)


model_appliers: Dict[str, ApplyModelMethod] = {
    'logistic_regression': apply_model_logistic_regression,
    'random_forest': apply_model_random_forest,
    'gradient_boosting_classifier': apply_model_gradient_boosting_classifier,
    'support_vector_machine': apply_model_support_vector_machine,
    'perceptron': apply_model_perceptron
}