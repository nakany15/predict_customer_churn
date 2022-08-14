import os
import logging
from src import churn_library as cls
import pytest
import os
import tempfile
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib

LOGGER = logging.getLogger(__name__)


def df_plugin():
    return None


def pytest_configure():
    pytest.df = df_plugin()


@pytest.fixture(scope='module')
def path():
    return "./data/bank_data.csv"


@pytest.fixture
def csv_path(tmpdir):
    raw_data_file_path = tmpdir.join("raw.csv")
    input_df = pd.DataFrame.from_dict(
        {
            'Attrition_Flag': [
                'Attrited Customer',
                'Attrited Customer',
                'Existing Customer',
                'Existing Customer',
            ],
            'Customer_Age': [
                58,
                55,
                55,
                42,
            ],
            'Gender': [
                'M',
                'F',
                'F',
                'M'],
            'Education_Level': [
                'Graduate',
                'College',
                'Graduate',
                'High School']})
    input_df.to_csv(raw_data_file_path)
    return raw_data_file_path


class TestImportData(object):
    def test_file_exist(self, path):
        try:
            pytest.df = cls.import_data(path)
            LOGGER.info("Testing test_file_exist: SUCCESS")
        except FileNotFoundError as err:
            LOGGER.error("Testing test_file_exist: The file wasn't found")
            pytest.df = pd.DataFrame()
            raise err

    def test_data_shape(self, csv_path):
        '''
        test data import - this example is completed for you to assist with the other test functions
        '''
        df = cls.import_data(csv_path)
        try:
            assert df.shape == [4, 4]
        except AssertionError as err:
            LOGGER.error(
                'test_data_shape: the imported data shape differs from csv')


    def test_target_exists(self, csv_path):
        df = cls.import_data(csv_path)
        try:
            assert 'Churn' in df.columns
        except AssertionError as err:
            LOGGER.error(
                "test_target_exists columns check: the DataFrame doesn't contain a target variable")
            raise err



class TestPerformEda(object):
    @pytest.fixture
    def input_df_fixt(self):
        input_df = pd.DataFrame.from_dict(
            {
                'Churn': [
                    1,
                    0,
                    0,
                    1,
                ],
                'Customer_Age': [
                    58,
                    55,
                    55,
                    42,
                ],
                'Gender': [
                    'M',
                    'F',
                    'F',
                    'M'],
                'Education_Level': [
                    'Graduate',
                    'College',
                    'Graduate',
                    'High School']})
        return input_df
    def test_eda_output(self, input_df_fixt, tmpdir):
        '''
        test perform eda function
        '''
        histograms_path = tmpdir.join('histograms.png')
        value_counts_path = tmpdir.join('value_counts.png')
        correlation_path = tmpdir.join('correlations.png')

        cls.perform_eda(
            input_df_fixt,
            tmpdir
        )
        try:
            assert os.path.isfile(histograms_path)
            assert os.path.isfile(value_counts_path)
            assert os.path.isfile(correlation_path)
        except AssertionError as err:
            LOGGER.error(
                "perform_eda function didn't produce expected image outputs")
            raise err


class TestEncoderHelper(object):
    @pytest.fixture
    def input_df_fixt(self):
        input_df = pd.DataFrame.from_dict(
            {
                'Churn': [
                    1,
                    0,
                    0,
                    1,
                ],
                'Customer_Age': [
                    58,
                    55,
                    55,
                    42,
                ],
                'Gender': [
                    'M',
                    'M',
                    'F',
                    'F'],
                'Education_Level': [
                    'Graduate',
                    'College',
                    'Graduate',
                    'High School']})
        return input_df

    def test_column_created(self, input_df_fixt):
        encoded_df = cls.encoder_helper(input_df_fixt,['Gender', 'Education_Level'], 'Churn')
        try:
            assert 'Gender_Churn' in encoded_df.columns
            assert 'Education_Level_Churn' in encoded_df.columns
        except AssertionError as err:
            LOGGER.error("encoder_helper function didn't create Gender_Churn")
            raise err

    def test_dataframe_shape(self, input_df_fixt):
        '''
        test encoder helper        
        '''
        encoded_df = cls.encoder_helper(input_df_fixt,['Gender', 'Education_Level'], 'Churn')
        try:
            assert encoded_df.shape[0] == input_df_fixt.shape[0]
            assert encoded_df.shape[1] == input_df_fixt.shape[1] + 2
        except AssertionError as err:
            LOGGER.error(
                "encoder_helper didn't produced expected output shape")
            raise err

    def test_education_encoded_values(self, input_df_fixt):
        encoded_df = cls.encoder_helper(input_df_fixt,['Gender', 'Education_Level'], 'Churn')
        try:
            assert encoded_df.loc[encoded_df['Education_Level']=='Graduate', 'Education_Level_Churn'].mean() == pytest.approx(0.5)
            assert encoded_df.loc[encoded_df['Education_Level']=='College', 'Education_Level_Churn'].mean() == pytest.approx(0)
            assert encoded_df.loc[encoded_df['Education_Level']=='High School', 'Education_Level_Churn'].mean() == pytest.approx(1)

        except AssertionError as err:
            LOGGER.error(
                "encoder_helper produced unexpected values of Education_Level_Churn")
            raise err

    def test_gender_encoded_values(self, input_df_fixt):
        encoded_df = cls.encoder_helper(input_df_fixt,['Gender', 'Education_Level'], 'Churn')
        try:
            assert encoded_df.loc[encoded_df['Gender']=='M', 'Gender_Churn'].unique() == pytest.approx(0.5)
            assert encoded_df.loc[encoded_df['Gender']=='F', 'Gender_Churn'].unique() == pytest.approx(0.5)
        except AssertionError as err:
            LOGGER.error(
                "the value of Gender_Churn must be 0.5 when Gender == 'M'")
            raise err

    def test_value_counts(self, input_df_fixt):
        encoded_df = cls.encoder_helper(input_df_fixt,['Gender', 'Education_Level'], 'Churn')

        try:
            assert len(
                encoded_df['Gender_Churn'].value_counts()) == 1
            assert len(
                encoded_df['Education_Level_Churn'].value_counts()) == len(
                encoded_df['Education_Level'].value_counts())
        except AssertionError as err:
            LOGGER.error(
                "number of unique values in encoded variables must be equel to original variables")
            raise err
    
    def test_no_encode(self, input_df_fixt):
        encoded_df = cls.encoder_helper(input_df_fixt,[], 'Churn')
        try:
            assert input_df_fixt.shape == encoded_df.shape
            assert set(input_df_fixt.columns) == set(encoded_df.columns)
        except AssertionError as err:
            LOGGER.error(
                "encoder_helper modified the shape of input DataFrame. that is invalid")
            raise err

class TestperformFeatureEngineering(object):
    '''
    test perform_feature_engineering
    '''
    @pytest.fixture(params=[['Gender'],
                            ['Education_Level',
                             'Marital_Status']])
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

    def test_column_length(self, mocker):
        mocker.patch(
            "src.churn_library.encoder_helper", 
            return_value=pd.DataFrame.from_dict(
            {
                'Churn': [
                    1,
                    0,
                    0,
                    1,
                ],
                'Customer_Age': [
                    58,
                    55,
                    55,
                    42,
                ],
                'Gender': [
                    'M',
                    'M',
                    'F',
                    'F'],
                'Gender_Churn': [
                    0.5,
                    0.5,
                    0.5,
                    0.5],
            })
        )
        df = pytest.df
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            pytest.df,
            ['Gender'],
            ['Customer_Age'],
            'Churn'
        )
        try:
            assert X_train.shape[1] == 2
            assert X_test.shape[1] == 2

        except AssertionError as err:
            LOGGER.error(
                "encoder_helper didn't produced expected number of columns")
            raise err

    def test_output_shapes(self, mocker):
        mocker.patch(
            "src.churn_library.encoder_helper", 
            return_value=pd.DataFrame.from_dict(
            {
                'Churn': [
                    1,
                    0,
                    0,
                    1,
                    0,
                    1,
                    1,
                    0,
                    1,
                    1,
                ],
                'Customer_Age': [
                    58,
                    55,
                    55,
                    42,
                    53,
                    58,
                    55,
                    55,
                    42,
                    53,
                ],
            })
        )
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            pytest.df,
            [],
            ['Customer_Age'],
            'Churn'
        )
        try:
            assert X_train.shape[0] == len(y_train)
            assert X_test.shape[0] == len(y_test)
            assert X_train.shape[0]  == 7
            assert X_test.shape[0] == 3
        except AssertionError as err:
            LOGGER.error(
                "encoder_helper didn't produced expected output shape")
            raise err

class TestCreateClassificationReport(object):
    def test_create_classification_report(self, mocker):
        fig = cls.create_classification_report(
            [1,0,0,1],
            [1,1,0,0],
            [1,0,0,1],
            [1,0,1,1],
            'Logistic Regression'
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        
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
            LOGGER.error("test_train_models didn't produce expected outputs")
            raise err
        # tear down
        temp_folder.cleanup()


if __name__ == "__main__":
    pass
