#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
import os
from sklearn import datasets, linear_model
from matplotlib import style
style.use("ggplot")
from sklearn.model_selection import cross_val_score, KFold
import plotly.graph_objs as go
from plotly.offline import plot


### Version 3.7.1 ###
#plotly.__version__

diab = datasets.load_diabetes()

# Linear regression

def plot_regression():
    for idx, col in enumerate(diab.data[:10]):
        diab_X = diab.data[:, np.newaxis, idx]

        # splitting into train and test sets
        X_train = diab_X[:-20]
        X_test  = diab_X[-20:]
        y_train = diab.target[:-20]
        y_test  = diab.target[-20:]
        
        regmodel = linear_model.LinearRegression()
        # train the model
        regmodel.fit(X_train, y_train)
        
        # Make predictions using the testing set
        y_pred = regmodel.predict(X_test)
#        print('Feature: ', diab.feature_names[idx].upper())
      
        plt.scatter(X_test, y_test, color = 'red')
        plt.plot(X_test, y_pred, color='blue', linewidth=2)
                          
        plt.xlabel('Value for {}'.format(diab.feature_names[idx].upper()))
        plt.ylabel('Predicted value for Y')
        plt.title('Regression model for {}'.format(diab.feature_names[idx].upper()))
        plt.xticks(())
        plt.yticks(())
        
        plt.savefig('./Figures/regr_{}.png'.format(diab.feature_names[idx].upper()))
        plt.show()
        print('Coefficients: %.2f' % regmodel.coef_)
        print('Mean Squared error: %.2f'
              % np.mean((regmodel.predict(X_test) - y_test)**2))
        print('R2 score: %.2f' % regmodel.score(X_test, y_test))
        print('RMSE: %.2f'% np.sqrt(np.mean((regmodel.predict(X_test) - y_test)**2)))
        plt.cla()
        plt.clf()
        plt.close()
        
def df_regr_analysis():
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 20)
    df = DataFrame({'--': ['COEFF', 'MSE', 'RMSE', 'R2'],
                    'AGE': [306.73, 5472.26, 73.97, -0.13],
                    'SEX': [59.78, 5501.91, 74.17, -0.14],
                    'BMI': [938.24, 2548.07, 50.48, 0.47],
                    'BP': [709.19, 4058.41, 63.71, 0.16],
                    'S1': [352.83, 5608.70, 74.89, -0.16],
                    'S2': [288.48, 5564.14, 74.59, -0.15],
                    'S3': [-647.35, 4538.34, 67.37, 0.06],
                    'S4': [701.13, 4850.82, 69.65, -0.00],
                    'S5': [900.39, 2923.34, 54.07, .39],
                    'S6': [630.54, 5265.50, 72.56, -0.09]
                    })
    df.set_index('--')
    print(df)        
          

'''

'''
def cross_validation():
    diabetes = datasets.load_diabetes()
    X = diabetes.data[:150]
    y = diabetes.target[:150]
    
    lasso = linear_model.Lasso(random_state=20)
    alphas = np.logspace(-4, -0.5, 30)
    
    scores = list()
    scores_std = list()
    
    n_folds = 3
    
    for alpha in alphas:
        lasso.alpha = alpha
        this_scores = cross_val_score(lasso, X, y, cv=n_folds, n_jobs=1)
        scores.append(np.mean(this_scores))
        scores_std.append(np.std(this_scores))
    
    scores, scores_std = np.array(scores), np.array(scores_std)
    print(scores, scores_std)
    p1 = go.Scatter(x=alphas, y=scores,
                mode='lines',
                line=dict(color='blue'),
                fill='tonexty'
               )

    std_error = scores_std / np.sqrt(n_folds)
    
    p2 = go.Scatter(x=alphas, y=scores + std_error, 
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    )
    
    p3 = go.Scatter(x=alphas, y=scores - std_error,
                    mode='lines',
                    line=dict(color='blue', dash='dash'),
                    fill='tonexty')
    
    line = go.Scatter(y=[np.max(scores), np.max(scores)],
                     x=[min(alphas), max(alphas)],
                     mode='lines',
                     line=dict(color='black', dash='dash', 
                               width=1),
                    )
    
    
    layout = go.Layout(xaxis=dict(title='alpha', type='log'),
                       yaxis=dict(title='CV score +/- std error'),
                       showlegend=False
                       )
    fig = go.Figure(data=[p2, p1, p3, line], layout=layout)

    plot(fig)
    
    lasso_cv = linear_model.LassoCV(alphas=alphas, random_state=0)
    k_fold = KFold(3)

    print('<<< Optimal alpha parameters to maximize scores 3-fold >>>')
    print('##########################################################')
          
    for k, (train, test) in enumerate(k_fold.split(X, y)):
        lasso_cv.fit(X[train], y[train])
        
        print('[fold {0}] alpha: {1:.5f}, score: {2:.5f}'.
              format(k, lasso_cv.alpha_, lasso_cv.score(X[test], y[test])))
    print('##########################################################')
    
'''
Select the best tuning parameters for KNN on the diabetes dataset
'''

def main():
    
    if not (os.path.exists('./Figures')):
        os.makedirs('./Figures')
    
    plot_regression() 
    cross_validation()
    #df_regr_analysis()
    cross_validation()
    
    
if __name__ == "__main__":
    main()




























