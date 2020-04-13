import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from itertools import chain, combinations
from scipy.optimize import nnls

# Constants
nbr_training_base_col = 6
nbr_testing_base_col = 7
min_percentage_ffill = 10


def get_log_returns(price_data, data_fill):
    """Remove line with empty data to ensure the data set is consistent and then compute log returns"""
    # Clear raw data
    if data_fill == 'Forward':
        price_data = price_data.fillna(method='ffill')
    elif data_fill == 'Backward':
        price_data = price_data.fillna(method='bfill')
    elif data_fill == 'Drop':
        # First use Forward Fill for the funds that publish the NAV every X days, then drop the dates with NaN
        missing_data = price_data.isnull().sum() * 100 / len(price_data)
        for items in missing_data.iteritems():
            if items[1] > min_percentage_ffill:
                price_data[items[0]] = price_data[items[0]].fillna(method='ffill')
                a = 1
        price_data.dropna()

    # Get log returns
    log_returns = np.log(price_data / price_data.shift(1))
    return log_returns.dropna()


def get_realized_vol(log_returns, time_windows):
    """Compute realized volatilities given a DF of returns"""
    # Get tickers list
    tickers_list = log_returns.columns
    # Get DateIndex
    dates_index = log_returns.index
    # Define MultiIndex for volatilities
    iterables = [time_windows, dates_index]
    multi_index_vol = pd.MultiIndex.from_product(iterables, names=['Time Window', 'Date'])
    multi_index_vol_structure = np.zeros((len(dates_index) * len(time_windows), len(tickers_list)))
    # Compute rolling volatilities
    for i in range(0, len(time_windows)):
        for j in range(0, len(tickers_list)):
            multi_index_vol_structure[i * len(dates_index):(i + 1) * len(dates_index), j] = \
                log_returns[log_returns.columns[j]].rolling(time_windows[i]).std() * np.sqrt(252)

    return pd.DataFrame(multi_index_vol_structure, index=multi_index_vol, columns=tickers_list).dropna()


def all_combinations(input_list):
    """Return all of the possible combinations for a given list"""
    unique_list = list(set(input_list))
    comb_list = list(chain.from_iterable(combinations(unique_list, x) for x in range(len(unique_list) + 1)))
    del comb_list[0]
    return comb_list


def get_regression_results(dependent_var_list, independent_var_combinations_list, rv, regression_type,
                           time_windows, d_cut):
    """Perform the regression and the Augmented Dickey-Fuller (ADF) test on all possible combinations"""
    # General variables
    formatted_ind_var = []
    nbr_max_ind_var = len(independent_var_combinations_list[-1])

    # Setup variables to build MultiIndex DataFrame
    df_structure = \
        np.zeros((len(time_windows) * len(dependent_var_list) * len(independent_var_combinations_list),
                  nbr_training_base_col + nbr_testing_base_col + (nbr_max_ind_var * 2) + 2))
    columns_list = get_columns_list(nbr_max_ind_var)

    # Regression analysis
    counter = 0
    # For each time window
    for t_wdw in time_windows:
        # For each dependent variable
        for dep_var in dependent_var_list:
            # Conduct regression analysis versus each possible combination
            for ind_var in independent_var_combinations_list:
                # Generate Combination column
                if (dep_var == dependent_var_list[0]) and (t_wdw == time_windows[0]):
                    formatted_ind_var.append('/'.join(ind_var))
                # Separate the dataset in pre/post cutoff for training and test purposes
                y, x = partition_dataset(rv, t_wdw, d_cut, dep_var, ind_var)

                # Conduct Regression and ADF Test
                # Ordinary Least Squares (OLS)
                if regression_type == 'OLS':
                    # Regression on training set
                    reg_model, r_squared, adj_r_squared, adf_results_training = run_ols(y[0], x[0])

                # Non-Negative Least Squares (NNLS)
                elif regression_type == 'NNLS':
                    reg_model, r_squared, adj_r_squared, adf_results_training = run_nnls(y[0], x[0])

                # Get stats on the testing dataset
                mse, min_error, max_error, adf_results_testing = get_test_model_stats(regression_type, reg_model,
                                                                                      x[1], y[1])

                # Structure results
                structure_regression_results(counter, regression_type, df_structure, reg_model, r_squared,
                                             adj_r_squared, adf_results_training, adf_results_testing,
                                             mse, min_error, max_error, nbr_max_ind_var)

                counter += 1

    iterables = [time_windows, dependent_var_list, formatted_ind_var]
    multi_index = \
        pd.MultiIndex.from_product(iterables,
                                   names=['Time Window', 'Dependent Variable', 'Independent Variables'])

    return pd.DataFrame(df_structure, index=multi_index, columns=columns_list)


def process_regression_results(regression_results, min_r_squared, confidence_level, adf_activate):
    """Remove the regressions that do not meet the minimum R-Squared condition and do not reject ADF's test H0 with the
    confidence level selected """
    # Round the values of R Squared and ADF Statistic to be able to spot regression that yielded the exact same result
    regression_results['R Square'] = regression_results['R Square'].round(decimals=5)
    regression_results['ADF Stat - Training'] = regression_results['ADF Stat - Training'].round(decimals=5)

    # Remove the regressions that yielded the exact same result
    filtered_regression_results = \
        regression_results.drop_duplicates(subset=['R Square', 'ADF Stat - Training'], keep='first')

    # Remove the regressions that do not meet minimum requirements
    if adf_activate:
        filtered_regression_results = \
            filtered_regression_results.loc[(filtered_regression_results['R Square'] > min_r_squared) &
                                            (filtered_regression_results['ADF Stat - Training'] <
                                             filtered_regression_results[str(confidence_level) + ' - Training']) &
                                            (filtered_regression_results['ADF Stat - Testing'] <
                                             filtered_regression_results[str(confidence_level) + ' - Testing'])
                                            ]
    else:
        filtered_regression_results = \
            filtered_regression_results.loc[(filtered_regression_results['R Square'] > min_r_squared)]

    # For those regression that met the requirements, organize them by dependent variable, sorted from highest to
    # lowest R-Squared
    filtered_regression_results.sort_values(['Time Window', 'Dependent Variable', 'R Square'],
                                            ascending=[False, True, False])
    return filtered_regression_results


def run_ols(y, x):
    """Run OLS regression"""
    # Regression
    reg_model_ols = sm.OLS(y, x).fit()
    # R Squared
    r_squared = reg_model_ols.rsquared
    adj_r_squared = reg_model_ols.rsquared_adj
    # ADF test
    adf_results = adfuller(reg_model_ols.resid)

    return reg_model_ols, r_squared, adj_r_squared, adf_results


def run_nnls(y, x):
    """Run NNLS regression"""
    # Regression
    x['const'] = -1
    reg_model_nnls = nnls(x, y)
    # If regression results did not include an intercept (positive value), change the constant to 1 and
    # run the regression again to this time get a positive coefficient for it
    if reg_model_nnls[0][0] == 0:
        x['const'] = 1
        reg_model_nnls = nnls(x, y)
    else:
        reg_model_nnls[0][0] *= -1
    # Compute model predicted Y
    nnls_predicted_y = np.sum(x * reg_model_nnls[0], 1)
    # Compute R-Squared
    nnls_r_squared = np.corrcoef(nnls_predicted_y, y)[0, 1] ** 2
    # Compute Adj. R-Squared
    nnls_adj_r_squared = nnls_r_squared - ((x.shape[1] - 1) / (x.shape[0] - x.shape[1])) * (1 - nnls_r_squared)
    # Compute Residuals
    nnls_resid = (nnls_predicted_y - y)
    # Conduct ADF test
    adf_results = adfuller(nnls_resid)

    return reg_model_nnls[0], nnls_r_squared, nnls_adj_r_squared, adf_results


def get_test_model_stats(regression_type, reg_model, x, y):
    """Test the model on the testing dataset and compute statistics"""
    if regression_type == 'OLS':
        predicted_y = reg_model.predict(x)
    elif regression_type == 'NNLS':
        predicted_y = np.sum(x * reg_model, 1)

    # Residuals
    residuals = (predicted_y - y)
    # Error Stats
    mse = np.average(residuals ** 2)
    min_error = np.min(residuals)
    max_error = np.max(residuals)

    # Conduct ADF test
    adf_results = adfuller(residuals)

    return mse, min_error, max_error, adf_results

def get_columns_list(nbr_max_ind_var):
    """Set up columns' names for regression DataFrame"""
    columns_list = ['R Square', 'Adj. R Square', 'ADF Stat - Training', '0.01 - Training', '0.05 - Training',
                    '0.1 - Testing', 'Intercept']
    for i in range(0, (nbr_max_ind_var * 2) + 1):
        if i < nbr_max_ind_var:
            columns_list.append('Coeff. ' + str(i + 1))
        elif i == nbr_max_ind_var:
            columns_list.append('Intercept T-Stat.')
        else:
            columns_list.append('T-Stat. ' + str(i - nbr_max_ind_var))
    columns_list.extend(['MSE - Testing', 'Min. Error - Testing', 'Max. Error - Testing', 'ADF Stat - Testing',
                         '0.01 - Testing', '0.05 - Testing', '0.1 - Testing'])

    return columns_list


def structure_regression_results(counter, regression_type, df_structure, reg_model, r_squared, adj_r_squared,
                                 adf_results_training, adf_results_testing, mse, min_error, max_error, nbr_max_ind_var):
    """Structure regression DataFrame data"""

    df_structure[counter, 0] = r_squared
    df_structure[counter, 1] = adj_r_squared
    df_structure[counter, 2] = adf_results_training[0]
    df_structure[counter, 3] = adf_results_training[4]['1%']
    df_structure[counter, 4] = adf_results_training[4]['5%']
    df_structure[counter, 5] = adf_results_training[4]['10%']

    if regression_type == 'OLS':
        col_index = nbr_training_base_col + len(reg_model.params)
        df_structure[counter, nbr_training_base_col:col_index] = reg_model.params
        col_index = nbr_training_base_col + nbr_max_ind_var + 1 + len(reg_model.tvalues)
        df_structure[counter, nbr_training_base_col + nbr_max_ind_var + 1:col_index] = reg_model.tvalues
    elif regression_type == 'NNLS':
        df_structure[counter, 5:5 + len(reg_model)] = reg_model

    col_index = nbr_training_base_col + (2 * nbr_max_ind_var) + 2
    df_structure[counter, col_index] = mse
    df_structure[counter, col_index + 1] = min_error
    df_structure[counter, col_index + 2] = max_error
    df_structure[counter, col_index + 3] = adf_results_testing[0]
    df_structure[counter, col_index + 4] = adf_results_testing[4]['1%']
    df_structure[counter, col_index + 5] = adf_results_testing[4]['5%']
    df_structure[counter, col_index + 6] = adf_results_testing[4]['10%']


def partition_dataset(rv, t_wdw, d_cut, dep_var, ind_var):
    """Split dataset in training and testing intervals"""
    idx = pd.IndexSlice
    y = [rv.loc[idx[t_wdw, rv.index.get_level_values('Date') < np.datetime64(d_cut)], dep_var],
         rv.loc[idx[t_wdw, rv.index.get_level_values('Date') > np.datetime64(d_cut)], dep_var]]
    x = [sm.add_constant(rv.loc[idx[t_wdw, rv.index.get_level_values('Date') < np.datetime64(d_cut)],
                                ind_var]),
         sm.add_constant(rv.loc[idx[t_wdw, rv.index.get_level_values('Date') > np.datetime64(d_cut)],
                                ind_var])]
    return y, x
