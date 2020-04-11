import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from itertools import chain, combinations
from scipy.optimize import nnls

# Constants
base_number_output_col = 5


def get_log_returns(price_data):
    """Function to remove line with empty data to ensure the data set is consistent and then compute log returns"""
    # Clear raw data
    price_data = price_data.dropna()
    # Get log returns
    log_returns = np.log(price_data / price_data.shift(1))
    return log_returns.dropna()


def get_realized_vol(log_returns, time_window):
    """Function to compute realized volatilities given a DF of returns"""
    # Get tickers list
    tickers_list = log_returns.columns
    # Get DateIndex
    dates_index = log_returns.index
    series_vol_structure = np.zeros((len(dates_index), len(tickers_list)))
    # Compute rolling volatilities
    for i in range(0,len(tickers_list)):
        series_vol_structure[:,i] = log_returns[log_returns.columns[i]].rolling(time_window).std() * np.sqrt(252)

    return pd.DataFrame(series_vol_structure, index=dates_index, columns=tickers_list).dropna()


def all_combinations(input_list):
    """Return all of the possible combinations for a given list"""
    unique_list = list(set(input_list))
    comb_list = list(chain.from_iterable(combinations(unique_list, x) for x in range(len(unique_list)+1)))
    del comb_list[0]
    return comb_list


def get_regression_results(dependent_var_list, independent_var_combinations_list, realized_vols, regression_type):
    """Function to perform the regression and the Augmented Dickey-Fuller (ADF) test on all possible combinations"""
    # Setup variables to build MultiIndex DataFrame
    counter = 0
    formated_ind_var = []
    multi_index_structure = np.zeros((len(dependent_var_list)*len(independent_var_combinations_list),
                                     base_number_output_col + len(independent_var_combinations_list[-1]) + 1))

    columns_list = ['R Square', 'ADF Stat', '0.01', '0.05', '0.1', 'Intercept']
    for i in range(0, len(independent_var_combinations_list[-1])):
        columns_list.append('Coeff. ' + str(i+1))

    # For each dependent variable
    for dependent_var in dependent_var_list:
        # Conduct regression analysis versus each possible combination
        for independent_var in independent_var_combinations_list:
            # Conduct OLS regression and ADF Test
            if dependent_var == dependent_var_list[0]:
                formated_ind_var.append('/'.join(independent_var))
            Y = realized_vols[dependent_var]
            X = sm.add_constant(realized_vols.loc[:, independent_var])
            # Regular OLS
            if regression_type == 'OLS':
                reg_model_ols = sm.OLS(Y, X).fit()
                # Conduct ADF test
                adf_results = adfuller(reg_model_ols.resid)
            # Non-Negative OLS
            elif regression_type == 'NNLS':
                X['const'] = -1
                reg_model_nnls, resid_nnls = nnls(X, Y)
                # If regression results did not include an intercept (positive value), change the constant to 1 and run
                # the regression again to this time get a positive coefficient for it
                if reg_model_nnls[0] == 0:
                    X['const'] = 1
                    reg_model_nnls, resid_nnls = nnls(X, Y)
                else:
                    reg_model_nnls[0] *= -1
                # Compute model predicted Y
                nnls_predicted_y = np.sum(X * reg_model_nnls, 1)
                # Compute R-Squared
                nnls_r_squared = np.corrcoef(nnls_predicted_y, Y)[0, 1] ** 2
                # Compute Residuals
                nnls_resid = (nnls_predicted_y - Y)
                # Conduct ADF test
                adf_results = adfuller(nnls_resid)

            # Structure results
            if regression_type == 'OLS':
                multi_index_structure[counter, 0] = reg_model_ols.rsquared
            elif regression_type == 'NNLS':
                multi_index_structure[counter, 0] = nnls_r_squared

            multi_index_structure[counter, 1] = adf_results[0]
            multi_index_structure[counter, 2] = adf_results[4]['1%']
            multi_index_structure[counter, 3] = adf_results[4]['5%']
            multi_index_structure[counter, 4] = adf_results[4]['10%']

            if regression_type == 'OLS':
                multi_index_structure[counter, 5:5+len(reg_model_ols.params)] = reg_model_ols.params
            elif regression_type == 'NNLS':
                multi_index_structure[counter, 5:5 + len(reg_model_nnls)] = reg_model_nnls

            counter += 1

    iterables = [dependent_var_list, formated_ind_var]
    multi_index_regression_results = pd.MultiIndex.from_product(iterables,
                                                                names=['Dependent Variable', 'Independent Variables'])

    return pd.DataFrame(multi_index_structure, index=multi_index_regression_results, columns=columns_list)


def process_regression_results(regression_results, min_r_squared, confidence_level):
    """Remove the regressions that do not meet the minimum R-Squared condition and do not reject ADF's test H0 with the
    confidence level selected """
    # Round the values of R Squared and ADF Statistic to be able to spot regression that yielded the exact same result
    regression_results['R Square'] = regression_results['R Square'].round(decimals=5)
    regression_results['ADF Stat'] = regression_results['ADF Stat'].round(decimals=5)
    # Remove the regressions that yielded the exact same result
    filtered_regression_results = regression_results.drop_duplicates(subset=['R Square', 'ADF Stat'], keep='first')
    # Remove the regressions that do not meet minimum requirements
    filtered_regression_results =\
        filtered_regression_results.loc[(filtered_regression_results['R Square'] > min_r_squared) &
                               (filtered_regression_results['ADF Stat'] < filtered_regression_results[str(confidence_level)])]
    # For those regression that met the requirements, organize them by dependent variable, sorted from highest to
    # lowest R-Squared
    filtered_regression_results.sort_values(['Dependent Variable', 'R Square'], ascending=[True, False])
    return filtered_regression_results