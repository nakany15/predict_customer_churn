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
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report
import os
os.environ['QT_QPA_PLATFORM']='offscreen'



def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    return pd.read_csv(pth)
    


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # translate Attrition_Flag to binary feature that value takes 1 when customer churn occures.
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    df = df.drop(columns = 
        [
            'Unnamed: 0',
            'CLIENTNUM',
            'Attrition_Flag',
        ]
    )
    quant_columns = list(df.select_dtypes(include = np.number).columns)
    cat_columns = list(df.select_dtypes(exclude = np.number).columns)

    fig, axes = plt.subplots(
        len(quant_columns), 
        1,
        figsize = (20, 10*len(quant_columns))
    )

    # plot histograms for quantitative features
    for i, (axis,col) in enumerate(zip(axes.flatten(),quant_columns)):
        df[col].hist(ax = axis)
    fig.savefig('./images/histgrams.png')

    fig, axes = plt.subplots(len(cat_columns), 1,figsize = (20,10*len(cat_columns)))

    # count plots for categorical features
    for i, (axis,col) in enumerate(zip(axes.flatten(),cat_columns)):
        df[col].value_counts('normalize').plot(kind = 'bar', ax = axis)
    fig.savefig('./images/value_counts.png')

    # plot heatmap that visualize correlation matrix
    fig = plt.figure(figsize=(20,10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    fig.savefig('./images/correlations.png')

def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    mean_df = pd.DataFrame()
    # target encoding for categorical features
    for col in category_lst:
        churn_means = (
            df.groupby(col)[response]
                .mean()
                .to_dict()
        )
        mean_values = df[col].map(churn_means)
        mean_df = (pd.concat(
                [mean_df,mean_values], 
                axis = 1
            )
            .rename(columns=
                {
                    col: f'{col}_Churn'
                }
            )
        )
    df = pd.concat([df, mean_df], axis = 1)
    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

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
    df['Churn'] = df[response].apply(lambda val: 0 if val == "Existing Customer" else 1)
    # perform target encoding
    df = encoder_helper(df, categorical_list, 'Churn')
    keep_cols = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
        'Income_Category_Churn', 'Card_Category_Churn'
    ]
    y = df['Churn']
    X = df[keep_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size= 0.3, 
        random_state=42
    )
    return X_train, X_test, y_train, y_test

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
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
    def create_classification_report(y_train, 
                                     y_test,
                                     y_train_preds, 
                                     y_test_preds,
                                     classifier
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
        fig = plt.figure(figsize = (10,5))
        #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
        plt.text(
            0.01, 
            1.25, 
            str(f'{classifier} Train'),
            {'fontsize': 10}, 
            fontproperties = 'monospace'
        )

        plt.text(
            0.01, 
            0.05, 
            str(classification_report(y_test, y_test_preds)), 
            {'fontsize': 10}, 
            fontproperties = 'monospace'
        ) # approach improved by OP -> monospace!

        plt.text(
            0.01, 
            0.6, 
            str(f'{classifier} Test'), 
            {'fontsize': 10}, 
            fontproperties = 'monospace'
        )

        plt.text(
            0.01, 
            0.7, 
            str(classification_report(y_train, y_train_preds)), 
            {'fontsize': 10}, 
            fontproperties = 'monospace'
        ) # approach improved by OP -> monospace!

        plt.axis('off')
        return fig
    
    # create classification report for Random Forest
    fig_rf = create_classification_report(
        y_train, 
        y_test, 
        y_train_preds_rf, 
        y_test_preds_rf,
        'Random Forest'
    )
    fig_rf.savefig('./images/classification_report_rf.png')

    # create classification report for Logistic Regression
    fig_lr = create_classification_report(
        y_train, 
        y_test, 
        y_train_preds_lr, 
        y_test_preds_lr,
        'Logisti Regression'
    )
    fig_lr.savefig('./images/classification_report_lr.png')




def feature_importance_plot(model, X_data, output_pth):
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
    names = [X_data.columns[i] for i in indices]
    
    plt.figure(figsize = (20,5))
    plt.title('Feature Importance')
    plt.ylabel('Importance')

    # feature importance bar plot
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation = 90)
    plt.savefig(output_pth)

def train_models(X_train, X_test, y_train, y_test):
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
    rfc = RandomForestClassifier(random_state = 42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # parameters for grid search
    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(
        estimator = rfc,
        param_grid = param_grid,
        cv = 5
    )
    # build models
    cv_rfc.fit(X_train, y_train)
    lrc .fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # create and save classification reports
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf
    )

    feature_importance_plot(
        cv_rfc.best_estimator_, 
        X_test, 
        './images/feature_importance_rf.png'
    )
    
    # ROC curve of logistic regression
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    # convine ROC curves of both logistic and random forest
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_, 
        X_test, 
        y_test, 
        ax=ax, 
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('./images/roc_curve.png')

    # save models
    joblib.dump(cv_rfc.best_estimator_,'./models/rfc_models.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == '__main__':
    df = import_data('./data/bank_data.csv')
    perform_eda(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df,
        'Attrition_Flag'
    )
    train_models(X_train, X_test, y_train, y_test)

