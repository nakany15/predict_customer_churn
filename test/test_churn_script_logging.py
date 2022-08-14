"""
This is a unit test program using pytest which test
functions developed in src/churn_library.py
Author: Yusuke Nakano
Date: 2022/8/10
"""


import os
import logging
from unittest.mock import MagicMock
import tempfile
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib
from src import churn_library as cls

LOGGER = logging.getLogger(__name__)


class TestImportData():
    """
    test class of unit test for import_data.
    """
    @pytest.fixture
    def path(self):
        """
        fixture function for test_file_exist
        Returns:
            csv file path
        """
        return "./data/bank_data.csv"

    @pytest.fixture
    def csv_path(self,tmpdir):
        """
        fixture function for unit test of import_data
        Returns:
            test data csv path located in temporary directory.
        """
        raw_data_file_path = tmpdir.join("raw.csv")
        input_df = pd.DataFrame.from_dict(
            {
                'Attrition_Flag': ['Attrited Customer', 'Attrited Customer', 'Existing Customer',],
                'Customer_Age': [58, 55, 55,],
                'Gender': ['M', 'F','F',],
                'Education_Level': ['Graduate','College','Graduate',]
            }
        )
        input_df.to_csv(raw_data_file_path, index = False)
        return raw_data_file_path

    def test_file_exist(self, path):
        """
        test for checking csv file exists.
        """
        try:
            df_import = cls.import_data(path)
            LOGGER.info("TestImportData test_file_exist: SUCCESS")
        except FileNotFoundError as err:
            LOGGER.error("TestImportData test_file_exist: The file wasn't found")
            raise err

    def test_data_shape(self, csv_path):
        '''
        testing shape of imported data is equal to csv file.
        '''
        df_import = cls.import_data(csv_path)
        try:
            assert df_import.shape == (3, 4)
            LOGGER.info("TestImportData test_data_shape: SUCCESS")
        except AssertionError as err:
            LOGGER.error(
                'TestImportData test_data_shape: the imported data shape differs from csv')
            LOGGER.info(df_import.shape)
            LOGGER.info(df_import.columns)
            raise err

    def test_target_exists(self, csv_path):
        """
        test if target variable 'Churn' is created properly by import_data function
        """
        df_import = cls.import_data(csv_path)
        try:
            assert 'Churn' in df_import.columns
            LOGGER.info("TestImportData test_target_exists: SUCCESS")
        except AssertionError as err:
            LOGGER.error(
                "TestImportData test_target_exists columns check: \
                    the DataFrame doesn't contain a target variable")
            raise err


class TestPerformEda():
    """
    a unit test class for testing perform_eda.
    """
    @pytest.fixture
    def input_df_fixt(self):
        """
        this is a fixture function to pass test data
        to unit test functions
        """
        input_df = pd.DataFrame.from_dict(
            {
                'Churn': [ 1, 0, 0, 1,],
                'Customer_Age': [58, 55, 55, 42,],
                'Gender': ['M','F','F','M'],
                'Education_Level': [ 'Graduate', 'College', 'Graduate','High School']
            }
        )
        return input_df

    def test_eda_output(self, input_df_fixt, tmpdir):
        '''
        test perform_eda save expected output in specified directory
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
            LOGGER.info("TestPerformEda test_eda_output: SUCCESS")
        except AssertionError as err:
            LOGGER.error(
                "TestPerformEda test_eda_output:\
                     function didn't produce expected image outputs")
            raise err


class TestEncoderHelper():
    """
    this is a unit test class for testing encoder_helper
    """
    @pytest.fixture
    def encode_df_fixt(self):
        """
        this is a fixture function passes a test data
        """
        input_df = pd.DataFrame.from_dict(
            {
                'Churn': [1, 0, 0, 1,],
                'Customer_Age': [58, 55, 55, 42,],
                'Gender': ['M', 'M', 'F', 'F'],
                'Education_Level': ['Graduate', 'College', 'Graduate', 'High School']
            }
        )
        return input_df

    def test_column_created(self, encode_df_fixt):
        """
        testing encoder_helper creates encoded columns from columns
        specified in the function argument.
        """
        encoded_df = cls.encoder_helper(
            encode_df_fixt,
            ['Gender', 'Education_Level'],
            'Churn'
        )
        try:
            assert 'Gender_Churn' in encoded_df.columns
            assert 'Education_Level_Churn' in encoded_df.columns
            LOGGER.info("TestEncoderHelper test_column_created: SUCCESS")
        except AssertionError as err:
            LOGGER.error("TestEncoderHelper test_column_created: \
                encoder_helper function didn't create Gender_Churn")
            raise err

    def test_dataframe_shape(self, encode_df_fixt):
        '''
        test that verifies encoder_helper produces DataFrame with expected data shape:
        - number of columns is equals to the number of columns of original DataFrame plus
          the number of encoded columns
        - the number of records is unchanged
        '''
        encoded_df = cls.encoder_helper(
            encode_df_fixt,
            ['Gender', 'Education_Level'],
            'Churn'
        )
        try:
            assert encoded_df.shape[0] == encode_df_fixt.shape[0]
            assert encoded_df.shape[1] == encode_df_fixt.shape[1] + 2
            LOGGER.info("TestEncoderHelper test_dataframe_shape: SUCCESS")
        except AssertionError as err:
            LOGGER.error(
                "TestEncoderHelper test_dataframe_shape: \
                    encoder_helper didn't produced expected output shape")
            raise err

    def test_education_encoded_values(self, encode_df_fixt):
        """
        test that verify encoder_helper encodes categorical features
        using target value(target encoding)
        """
        encoded_df = cls.encoder_helper(
            encode_df_fixt, [
                'Gender', 'Education_Level'], 'Churn')
        try:
            assert encoded_df.loc[encoded_df['Education_Level'] == 'Graduate',
                                  'Education_Level_Churn'].mean() == pytest.approx(0.5)
            assert encoded_df.loc[encoded_df['Education_Level'] ==
                                  'College', 'Education_Level_Churn'].mean() == pytest.approx(0)
            assert encoded_df.loc[encoded_df['Education_Level'] ==
                                  'High School', 'Education_Level_Churn'].mean() == pytest.approx(1)
            LOGGER.info("TestEncoderHelper test_education_encoded_values: SUCCESS")
        except AssertionError as err:
            LOGGER.error(
                "TestEncoderHelper test_education_encoded_values:\
                    encoder_helper produced unexpected values of Education_Level_Churn")
            raise err

    def test_no_encode(self, encode_df_fixt):
        """
        test that checks encoder_helper produces no error and returns
        DataFrame that is idendical to the input DataFrame
        """
        encoded_df = cls.encoder_helper(encode_df_fixt, [], 'Churn')
        try:
            assert encode_df_fixt.shape == encoded_df.shape
            assert set(encode_df_fixt.columns) == set(encoded_df.columns)
            LOGGER.info("TestEncoderHelper test_no_encode: SUCCESS")
        except AssertionError as err:
            LOGGER.error(
                "TestEncoderHelper test_no_encode: \
                    encoder_helper modified the shape of input DataFrame. that is invalid")
            raise err


class TestperformFeatureEngineering():
    '''
    test class for perform_feature_engineering.
    perform_feature_engineering firstly calls encoder_helper.
    virtually, the input DataFrame of the function is the output
    DataFrame of encoder_helper.
    therefore, mocker patch is assigned to encoder_helper function
    to fix the output DataFrame of it.
    '''
    def test_column_length(self, mocker):
        """
        testing for column length of output DataFrames
        """
        # mocker patch to fix the output DataFrame of encoder_helper
        mocker.patch(
            "src.churn_library.encoder_helper",
            return_value=pd.DataFrame.from_dict(
                {
                    'Churn': [1, 0, 0, 1,],
                    'Customer_Age': [58, 55, 55, 42,],
                    'Gender': ['M', 'M', 'F', 'F'],
                    'Gender_Churn': [0.5, 0.5, 0.5, 0.5],
                }
            )
        )
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
            pd.DataFrame(),
            ['Gender'],
            ['Customer_Age'],
            'Churn'
        )
        try:
            assert x_train.shape[1] == 2
            assert x_test.shape[1] == 2
            LOGGER.info("TestperformFeatureEngineering test_column_length: SUCCESS")
        except AssertionError as err:
            LOGGER.error(
                "TestperformFeatureEngineering test_column_length: \
                    perform_feature_engineering didn't produced expected number of columns")
            raise err

    def test_train_test_record_shape(self, mocker):
        """
        test if the number of training data and test data are
        70% and 30% of input DataFrame respectively.

        """
        # mocker patch to fix the output DataFrame of encoder_helper
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
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
            pd.DataFrame(),
            [],
            ['Customer_Age'],
            'Churn'
        )
        try:
            assert x_train.shape[0] == len(y_train)
            assert x_test.shape[0] == len(y_test)
            assert x_train.shape[0] == 7
            assert x_test.shape[0] == 3
            LOGGER.info("TestperformFeatureEngineering test_train_test_record_shape: SUCCESS")
        except AssertionError as err:
            LOGGER.error(
                "TestperformFeatureEngineering test_train_test_record_shape: \
                    perform_feature_enggineering didn't produced expected output shape")
            raise err


class TestCreateClassificationReport():
    """
    test class for create_classification_report
    """
    def test_create_classification_report(self):
        """
        test if the function produces expected output type:
        expected output is the instance of matplotlib.figure.Figure
        """
        fig = cls.create_classification_report(
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            'Logistic Regression'
        )
        try:
            assert isinstance(fig, matplotlib.figure.Figure)
            LOGGER.info("TestCreateClassificationReport test_create_classification_report: SUCCESS")
        except AssertionError as err:
            LOGGER.error(
                "TestCreateClassificationReport test_create_classification_report: \
                    create_classification_report didn't return expected object type")
            raise err


class TestFeatureImportancePlot():
    """
    test class for feature_importance_plot
    """
    @pytest.fixture
    def input_x_fixt(self,):
        """
        fixture function to pass test DataFrame
        """
        input_x = pd.DataFrame.from_dict(
            {
                'Customer_Age': [1, 2, 3],
                'Gender': [1, 0, 1],
                'Salary': [400, 250, 500]
            }
        )
        return input_x

    def test_plot_exist(self, input_x_fixt, tmpdir):
        """
        test for checking the function saves feature_importances_rf.png
        to the directoory specified in a function argument.
        """
        mock_model = MagicMock(feature_importances_=np.array([1, 4, 2]))
        cls.feature_importance_plot(mock_model, input_x_fixt, tmpdir)
        try:
            assert os.path.isfile(tmpdir.join('feature_importances_rf.png'))
            LOGGER.info("TestFeatureImportancePlot test_plot_exist: SUCCESS")
        except AssertionError as err:
            LOGGER.error(
                "TestFeatureImportancePlot test_plot_exist: \
                    feature_importance_plot didn't save expected image file")
            raise err


class TestTrainModels():
    '''
    test train_models
    '''
    @pytest.fixture
    def param_grid_fixt(self):
        """
        fixture function to pass grid search parameter dictionary
        """
        params = {
            'n_estimators': [10, 20],
            'max_features': ['sqrt'],
            'max_depth': [4, 5],
            'criterion': ['gini']
        }
        return params

    @pytest.fixture
    def sample_data_fixt(self):
        """
        fixture function to pass test DataFrame
        """
        x_train_test, y_train_test = make_classification(
            random_state=12,
            n_samples=1000,
            n_features=5,
            n_redundant=0,
            n_informative=1,
            n_clusters_per_class=1,
            n_classes=2
        )
        x_train_test = pd.DataFrame(x_train_test)
        x_train, x_test, y_train, y_test = train_test_split(
            x_train_test,
            y_train_test,
            test_size=0.3,
            random_state=22
        )
        return [x_train, x_test, y_train, y_test]

    def test_outputs_exists(
            self,
            tmpdir,
            param_grid_fixt,
            sample_data_fixt,
            ):
        """
        test if train_model saves expected outputs in
        the directories specified in function arguments.
        """
        temp_folder = tempfile.TemporaryDirectory()

        classification_report_path_rf = tmpdir.join(
            'classification_report_lr.png')
        classification_report_path_lr = tmpdir.join(
            'classification_report_rf.png')
        importance_path_rf = tmpdir.join('feature_importances_rf.png')
        roc_path = tmpdir.join('roc_curve.png')
        model_path_lr = os.path.join(temp_folder.name, 'logistic_model.pkl')
        model_path_rf = os.path.join(temp_folder.name, 'rfc_model.pkl')
        x_train, x_test, y_train, y_test = sample_data_fixt

        cls.train_models(
            x_train,
            x_test,
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
            LOGGER.info("TestTrainModels test_outputs_exists: SUCCESS")
        except AssertionError as err:
            LOGGER.error("TestTrainModels test_outputs_exists:\
                train_models didn't produce expected outputs")
            LOGGER.info(os.listdir(tmpdir))
            raise err
        # tear down
        temp_folder.cleanup()


if __name__ == "__main__":
    pass
