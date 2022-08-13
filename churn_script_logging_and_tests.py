import os
import logging
import churn_library as cls
import pytest

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def df_plugin():
	return None

def pytest_configure():
	pytest.df = df_plugin()


@pytest.fixture(scope='module')
def path():
	return "./data/bank_data.csv"

def test_import_data(path):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		pytest.df = cls.import_data(path)
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err
	try:
		assert pytest.df.shape[0] > 0
		assert pytest.df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda():
	'''
	test perform eda function
	'''
	df = pytest.df
	try:
		assert 'Attrition_Flag' in df.columns
		assert 'Unnamed: 0' in df.columns
		assert 'CLIENTNUM' in df.columns
	except AssertionError as err:
		logging.error("test_eda columns check: the DataFrame doesn't contain a target variable")
	cls.perform_eda(df)



def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	'''


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''


def test_train_models(train_models):
	'''
	test train_models
	'''


if __name__ == "__main__":
	pass








