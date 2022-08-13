# library doc string
"""
These modules are implemented to predict customer churn.
Models are created by following steps.
1. import data from csv
2. perform EDA
3. feature engineering
4. build machine learning model using Logistic Regression and Random Forest.
5. evaluate model performances and save models.
"""

# import libraries
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df_train = pd.read_csv(pth)
    df_train['Churn'] = df_train['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    df_train.drop(columns=['Attrition_Flag'])
    return df_train


def perform_eda(df_train, out_plot_path,):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # translate Attrition_Flag to binary feature that value takes 1 when
    # customer churn occures.
    if 'Unnamed: 0' in df_train.columns:
        df_train = df_train.drop(columns=['Unnamed: 0'])
    if 'CLIENTNUM' in df_train.columns:
        df_train = df_train.drop(columns=['CLIENTNUM'])
    quant_columns = list(df_train.select_dtypes(include=np.number).columns)
    cat_columns = list(df_train.select_dtypes(exclude=np.number).columns)

    fig, axes = plt.subplots(
        len(quant_columns),
        1,
        figsize=(20, 10 * len(quant_columns))
    )

    # plot histograms for quantitative features
    for axis, col in zip(axes.flatten(), quant_columns):
        df_train[col].hist(ax=axis)
    fig.savefig(os.path.join(out_plot_path, 'histograms.png'))

    fig, axes = plt.subplots(
        len(cat_columns), 1, figsize=(
            20, 10 * len(cat_columns)))

    # count plots for categorical features
    for axis, col in zip(axes.flatten(), cat_columns):
        df_train[col].value_counts('normalize').plot(kind='bar', ax=axis)
    fig.savefig(os.path.join(out_plot_path, 'value_counts.png'))

    # plot heatmap that visualize correlation matrix
    fig = plt.figure(figsize=(20, 10))
    sns.heatmap(df_train.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    fig.savefig(os.path.join(out_plot_path, 'correlations.png'))


def encoder_helper(df_train, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument
                      that could be used for naming variables or index y column

    output:
            df: pandas dataframe with new columns for
    '''
    mean_df = pd.DataFrame()
    # target encoding for categorical features
    for col in category_lst:
        churn_means = (
            df_train.groupby(col)[response]
            .mean()
            .to_dict()
        )
        mean_values = df_train[col].map(churn_means)
        mean_df = (pd.concat(
            [mean_df, mean_values],
            axis=1
        )
            .rename(columns={
                    col: f'{col}_Churn'
                    }
                    )
        )
    df_train = pd.concat([df_train, mean_df], axis=1)
    return df_train


def perform_feature_engineering(df_train, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument
                        that could be used for naming variables or index y column

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    categorical_list = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    # create target variable that takes 1 if customer churn occures
    # perform target encoding
    df_train = encoder_helper(df_train, categorical_list, response)
    keep_cols = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn'
    ]
    y_train_test = df_train[response]
    x_train_test = df_train[keep_cols]
    x_train, x_test, y_train, y_test = train_test_split(
        x_train_test,
        y_train_test,
        test_size=0.3,
        random_state=42
    )
    return x_train, x_test, y_train, y_test


def create_classification_report(y_train,
                                    y_test,
                                    y_train_preds,
                                    y_test_preds,
                                    classifier,
                                    ):
    """
    create classification report for training and testing results
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds:  training predictions
            y_test_preds:   test predictions
            classifier:  classifier name used for prediction.
                            This string value is used for titles of the report.
    output:
            pyplot.figure: classification report figure
    """
    fig = plt.figure(figsize=(10, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old
    # approach
    plt.text(
        0.01,
        1.25,
        str(f'{classifier} Train'),
        {'fontsize': 10},
        fontproperties='monospace'
    )

    plt.text(
        0.01,
        0.05,
        str(classification_report(y_test, y_test_preds)),
        {'fontsize': 10},
        fontproperties='monospace'
    )  # approach improved by OP -> monospace!

    plt.text(
        0.01,
        0.6,
        str(f'{classifier} Test'),
        {'fontsize': 10},
        fontproperties='monospace'
    )

    plt.text(
        0.01,
        0.7,
        str(classification_report(y_train, y_train_preds)),
        {'fontsize': 10},
        fontproperties='monospace'
    )  # approach improved by OP -> monospace!

    plt.axis('off')
    return fig

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                out_dir,
                                ):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    # create classification report for Logistic Regression
    fig_lr = create_classification_report(
        y_train,
        y_test,
        y_train_preds_lr,
        y_test_preds_lr,
        'Logisti Regression'
    )
    fig_lr.savefig(os.path.join(out_dir, 'classification_report_lr.png'))

    # create classification report for Random Forest
    fig_rf = create_classification_report(
        y_train,
        y_test,
        y_train_preds_rf,
        y_test_preds_rf,
        'Random Forest'
    )
    fig_rf.savefig(os.path.join(out_dir, 'classification_report_rf.png'))


def feature_importance_plot(model, x_data, output_dir):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title('Feature Importance')
    plt.ylabel('Importance')

    # feature importance bar plot
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(os.path.join(output_dir, 'feature_importances_rf.png'))


def train_models(x_train,
                 x_test,
                 y_train,
                 y_test,
                 param_grid,
                 out_plot_dir,
                 out_model_dir,
                 ):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    cv_rfc = GridSearchCV(
        estimator = RandomForestClassifier(random_state=42),
        param_grid = param_grid,
        cv=5
    )

    # build models
    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # create and save classification reports
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
        out_plot_dir,
    )

    feature_importance_plot(
        cv_rfc.best_estimator_,
        x_test,
        out_plot_dir,
    )

    # ROC curve of logistic regression
    lrc_plot = plot_roc_curve(lrc, x_test, y_test)
    # convine ROC curves of both logistic and random forest
    plt.figure(figsize=(15, 8))
    ax_roc = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        ax=ax_roc,
        alpha=0.8)
    lrc_plot.plot(ax=ax_roc, alpha=0.8)
    plt.savefig(os.path.join(out_plot_dir, 'roc_curve.png'))
    #del rfc_disp

    # save models
    joblib.dump(
        lrc,
        os.path.join(out_model_dir, 'logistic_model.pkl')
    )
    joblib.dump(
        cv_rfc.best_estimator_,
        os.path.join(out_model_dir, 'rfc_model.pkl')
    )


if __name__ == '__main__':
    df = import_data('./data/bank_data.csv')
    perform_eda(
        df,
        './images',
    )
    X_train_churn, X_test_churn, y_train_churn, y_test_churn = perform_feature_engineering(
        df, 'Churn')
    param_grid_churn = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    train_models(
        X_train_churn,
        X_test_churn,
        y_train_churn,
        y_test_churn,
        param_grid_churn,
        './images',
        './models'
    )
