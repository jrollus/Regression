import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from itertools import chain, combinations
from scipy.optimize import nnls

# Constants
nbr_out_col = 6
min_percentage_ffill = 10

def get_log_returns(price_data, data_fill):
    """Function to remove line with empty data to ensure the data set is consistent and then compute log returns"""
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
    """Function to compute realized volatilities given a DF of returns"""
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
            multi_index_vol_structure[i*len(dates_index):(i+1)*len(dates_index), j] =\
                log_returns[log_returns.columns[j]].rolling(time_windows[i]).std() * np.sqrt(252)

    return pd.DataFrame(multi_index_vol_structure, index=multi_index_vol, columns=tickers_list).dropna()


def all_combinations(input_list):
    """Return all of the possible combinations for a given list"""
    unique_list = list(set(input_list))
    comb_list = list(chain.from_iterable(combinations(unique_list, x) for x in range(len(unique_list)+1)))
    del comb_list[0]
    return comb_list


def get_regression_results(dependent_var_list, independent_var_combinations_list, realized_vols, regression_type,
                           time_windows):
    """Function to perform the regression and the Augmented Dickey-Fuller (ADF) test on all possible combinations"""
    # Setup variables to build MultiIndex DataFrame
    counter = 0
    formated_ind_var = []
    idx = pd.IndexSlice
    nbr_max_ind_var = len(independent_var_combinations_list[-1])
    multi_index_structure = \
        np.zeros((len(time_windows) * len(dependent_var_list) * len(independent_var_combinations_list),
                  nbr_out_col + (nbr_max_ind_var * 2) + 2))

    columns_list = ['R Square', 'Adj. R Square', 'ADF Stat', '0.01', '0.05', '0.1', 'Intercept']

    for i in range(0, (nbr_max_ind_var*2) + 1):
        if i < nbr_max_ind_var:
            columns_list.append('Coeff. ' + str(i+1))
        elif i == nbr_max_ind_var:
            columns_list.append('Intercept T-Stat.')
        else:
            columns_list.append('T-Stat. ' + str(i - nbr_max_ind_var))

    # For each time window
    for time_window in time_windows:
        # For each dependent variable
        for dependent_var in dependent_var_list:
            # Conduct regression analysis versus each possible combination
            for independent_var in independent_var_combinations_list:
                # Conduct OLS regression and ADF Test
                if (dependent_var == dependent_var_list[0]) and (time_window == time_windows[0]):
                    formated_ind_var.append('/'.join(independent_var))
                y = realized_vols.loc[idx[time_window, :], dependent_var]
                x = sm.add_constant(realized_vols.loc[idx[time_window, :], independent_var])
                # Regular OLS
                if regression_type == 'OLS':
                    reg_model_ols = sm.OLS(y, x).fit()
                    # Conduct ADF test
                    adf_results = adfuller(reg_model_ols.resid)
                # Non-Negative OLS
                elif regression_type == 'NNLS':
                    x['const'] = -1
                    reg_model_nnls, resid_nnls = nnls(x, y)
                    # If regression results did not include an intercept (positive value), change the constant to 1 and
                    # run the regression again to this time get a positive coefficient for it
                    if reg_model_nnls[0] == 0:
                        x['const'] = 1
                        reg_model_nnls, resid_nnls = nnls(x, y)
                    else:
                        reg_model_nnls[0] *= -1
                    # Compute model predicted Y
                    nnls_predicted_y = np.sum(x * reg_model_nnls, 1)
                    # Compute R-Squared
                    nnls_r_squared = np.corrcoef(nnls_predicted_y, y)[0, 1] ** 2
                    # Compute Adj. R-Squared
                    nnls_adj_r_squared = nnls_r_squared - ((x.shape[1]-1)/(x.shape[0]-x.shape[1])) * (1-nnls_r_squared)
                    # Compute Residuals
                    nnls_resid = (nnls_predicted_y - y)
                    # Conduct ADF test
                    adf_results = adfuller(nnls_resid)

                # Structure results
                if regression_type == 'OLS':
                    multi_index_structure[counter, 0] = reg_model_ols.rsquared
                    multi_index_structure[counter, 1] = reg_model_ols.rsquared_adj
                elif regression_type == 'NNLS':
                    multi_index_structure[counter, 0] = nnls_r_squared
                    multi_index_structure[counter, 1] = nnls_adj_r_squared

                multi_index_structure[counter, 2] = adf_results[0]
                multi_index_structure[counter, 3] = adf_results[4]['1%']
                multi_index_structure[counter, 4] = adf_results[4]['5%']
                multi_index_structure[counter, 5] = adf_results[4]['10%']

                if regression_type == 'OLS':
                    col_index = nbr_out_col + len(reg_model_ols.params)
                    multi_index_structure[counter, nbr_out_col:col_index] = reg_model_ols.params
                    col_index = nbr_out_col + nbr_max_ind_var + 1 + len(reg_model_ols.tvalues)
                    multi_index_structure[counter, nbr_out_col + nbr_max_ind_var + 1:col_index] = reg_model_ols.tvalues
                elif regression_type == 'NNLS':
                    multi_index_structure[counter, 5:5 + len(reg_model_nnls)] = reg_model_nnls

                counter += 1

    iterables = [time_windows, dependent_var_list, formated_ind_var]
    multi_index_regression_results =\
        pd.MultiIndex.from_product(iterables,
                                   names=['Time Window', 'Dependent Variable', 'Independent Variables'])

    return pd.DataFrame(multi_index_structure, index=multi_index_regression_results, columns=columns_list)


def process_regression_results(regression_results, min_r_squared, confidence_level, adf_activate):
    """Remove the regressions that do not meet the minimum R-Squared condition and do not reject ADF's test H0 with the
    confidence level selected """
    # Round the values of R Squared and ADF Statistic to be able to spot regression that yielded the exact same result
    regression_results['R Square'] = regression_results['R Square'].round(decimals=5)
    regression_results['ADF Stat'] = regression_results['ADF Stat'].round(decimals=5)

    # Remove the regressions that yielded the exact same result
    filtered_regression_results = regression_results.drop_duplicates(subset=['R Square', 'ADF Stat'], keep='first')

    # Remove the regressions that do not meet minimum requirements
    if adf_activate:
        filtered_regression_results =\
            filtered_regression_results.loc[(filtered_regression_results['R Square'] > min_r_squared) &
                                            (filtered_regression_results['ADF Stat'] <
                                             filtered_regression_results[str(confidence_level)])]
    else:
        filtered_regression_results = \
            filtered_regression_results.loc[(filtered_regression_results['R Square'] > min_r_squared)]

    # For those regression that met the requirements, organize them by dependent variable, sorted from highest to
    # lowest R-Squared
    filtered_regression_results.sort_values(['Time Window', 'Dependent Variable', 'R Square'],
                                            ascending=[False, True, False])
    return filtered_regression_results
