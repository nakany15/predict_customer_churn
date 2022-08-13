import os
import logging
from src import churn_library as cls
import pytest
import os
import tempfile
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def df_plugin():
    return None


def pytest_configure():
    pytest.df = df_plugin()


@pytest.fixture(scope='module')
def path():
    return "./data/bank_data.csv"


class TestImportData(object):
    def test_file_exist(self, path):
        try:
            pytest.df = cls.import_data(path)
            logging.info("Testing import_data: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing import_eda: The file wasn't found")
            raise err

    def test_record_greater_than_zero(self):
        '''
        test data import - this example is completed for you to assist with the other test functions
        '''
        df = pytest.df
        try:
            assert df.shape[0] > 0
            assert df.shape[1] > 0
        except AssertionError as err:
            logging.error(
                "Testing import_data: The file doesn't appear to have rows and columns")
            raise err

    def test_target_exists(self):
        df = pytest.df
        try:
            assert 'Churn' in df.columns
        except AssertionError as err:
            logging.error(
                "test_eda columns check: the DataFrame doesn't contain a target variable")
            raise err


class TestPerformEda(object):

    def test_eda_output(self, tmpdir):
        '''
        test perform eda function
        '''
        df = pytest.df
        histograms_path = tmpdir.join('histograms.png')
        value_counts_path = tmpdir.join('value_counts.png')
        correlation_path = tmpdir.join('correlations.png')

        cls.perform_eda(
            df,
            tmpdir
        )
        try:
            assert os.path.isfile(histograms_path)
            assert os.path.isfile(value_counts_path)
            assert os.path.isfile(correlation_path)
        except AssertionError as err:
            logging.error(
                "perform_eda function didn't produce expected image outputs")
            raise err


class TestEncoderHelper(object):
    @pytest.fixture
    def encoder_fixt(self):
        df = pytest.df
        df_out = cls.encoder_helper(
            df, ['Income_Category', 'Education_Level'], 'Churn')
        return df_out

    def test_column_created(self, encoder_fixt):
        try:
            assert 'Income_Category_Churn' in encoder_fixt.columns
            assert 'Education_Level_Churn' in encoder_fixt.columns
        except AssertionError as err:
            logging.error("encoder_helper function didn't create Gender_Churn")
            raise err

    def test_dataframe_shape(self, encoder_fixt):
        '''
        test encoder helper
        '''
        df = pytest.df
        try:
            assert encoder_fixt.shape[0] == df.shape[0]
            assert encoder_fixt.shape[1] == df.shape[1] + 2
        except AssertionError as err:
            logging.error(
                "encoder_helper didn't produced expected output shape")
            raise err

    def test_value_counts(self, encoder_fixt):
        try:
            assert len(
                encoder_fixt['Income_Category_Churn'].value_counts()) == len(
                encoder_fixt['Income_Category'].value_counts())
            assert len(
                encoder_fixt['Education_Level_Churn'].value_counts()) == len(
                encoder_fixt['Education_Level'].value_counts())
        except AssertionError as err:
            logging.error(
                "number of unique values in encoded variables must be equel to original variables")
            raise err


class TestperformFeatureEngineering(object):
    '''
    test perform_feature_engineering
    '''
    @pytest.fixture(params=[['Gender'],
                            ['Education_Level',
                             'Marital_Status'],
                            ['Income_Category',
                             'Card_Category',
                             'Education_Level']])
    def encoding_category_fixt(self, request):
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            pytest.df,
            request.param,
            [
                'Customer_Age',
                'Dependent_count',
                'Months_on_book',
            ],
            'Churn',
        )
        return [X_train, X_test, y_train, y_test, len(request.param)]

    def test_column_length(self, encoding_category_fixt):
        df = pytest.df
        X_train, X_test, y_train, y_test, cat_len = encoding_category_fixt
        try:
            assert X_train.shape[1] == 3 + cat_len
            assert X_test.shape[1] == 3 + cat_len

        except AssertionError as err:
            logging.error(
                "encoder_helper didn't produced expected number of columns")
            raise err

    def test_output_shapes(self, encoding_category_fixt):

        df = pytest.df
        X_train, X_test, y_train, y_test, cat_len = encoding_category_fixt
        try:
            assert X_train.shape[1] == 3 + cat_len
            assert X_test.shape[1] == 3 + cat_len
            assert X_train.shape[0] == len(y_train)
            assert X_test.shape[0] == len(y_test)
            assert X_train.shape[0] / \
                df.shape[0] == pytest.approx(0.7, rel=1e-3)
            assert X_test.shape[0] / \
                df.shape[0] == pytest.approx(0.3, rel=1e-3)

        except AssertionError as err:
            logging.error(
                "encoder_helper didn't produced expected output shape")
            raise err

    def test_record_not_changes(self, encoding_category_fixt):
        df = pytest.df
        X_train, X_test, y_train, y_test, cat_len = encoding_category_fixt
        try:
            assert len(X_train) + len(X_test) == len(df)
            assert len(y_train) + len(y_test) == len(df)
        except AssertionError as err:
            logging.error(
                "total number of train and test must be equal to input DataFrame")
            raise err


class TestTrainModels(object):
    '''
    test train_models
    '''
    @pytest.fixture
    def param_grid_fixt(self):
        params = {
            'n_estimators': [10, 20],
            'max_features': ['sqrt'],
            'max_depth': [4, 5],
            'criterion': ['gini']
        }
        return params

    @pytest.fixture
    def sample_data_fixt(self):
        X, y = make_classification(
            random_state=12,
            n_samples=1000,
            n_features=5,
            n_redundant=0,
            n_informative=1,
            n_clusters_per_class=1,
            n_classes=2
        )
        X = pd.DataFrame(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=22
        )
        return [X_train, X_test, y_train, y_test]

    def test_outputs_exists(self, tmpdir, param_grid_fixt, sample_data_fixt):
        temp_folder = tempfile.TemporaryDirectory()

        classification_report_path_rf = tmpdir.join(
            'classification_report_lr.png')
        classification_report_path_lr = tmpdir.join(
            'classification_report_rf.png')
        importance_path_rf = tmpdir.join('feature_importances_rf.png')
        roc_path = tmpdir.join('roc_curve.png')
        model_path_lr = os.path.join(temp_folder.name, 'logistic_model.pkl')
        model_path_rf = os.path.join(temp_folder.name, 'rfc_model.pkl')
        X_train, X_test, y_train, y_test = sample_data_fixt

        cls.train_models(
            X_train,
            X_test,
            y_train,
            y_test,
            param_grid_fixt,
            tmpdir,
            temp_folder.name
        )
        try:
            assert os.path.isfile(classification_report_path_rf)
            assert os.path.isfile(classification_report_path_lr)
            assert os.path.isfile(importance_path_rf)
            assert os.path.isfile(roc_path)
            assert os.path.isfile(model_path_lr)
            assert os.path.isfile(model_path_rf)
        except AssertionError as err:
            logging.error("test_train_models didn't produce expected outputs")
            raise err
        # tear down
        temp_folder.cleanup()


if __name__ == "__main__":
    pass