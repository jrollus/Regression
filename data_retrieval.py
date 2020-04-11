import gui as gu
import numpy as np
from xbbg import blp


def get_relevant_data(tickers_list, date_start, date_end, data_source):
    """Get financial data from Bloomberg or Typhoon for a given set of underlying and type of data"""
    if data_source == 'Bloomberg':
        selected_data = blp.bdh(tickers=tickers_list, flds=['PX_LAST'],
                                start_date=date_start, end_date=date_end, adjust='all')

        if not selected_data.empty:
            selected_data.columns = selected_data.columns.droplevel(1)

    return selected_data


def check_data_retrieval_error(price_data, dependent_var_list, independent_var_list, data_source):
    """Check whether there was an error retrieving data"""
    # Check whether data was retrieved successfully:
    tickers_list = dependent_var_list + independent_var_list
    if data_source == 'Bloomberg':
        if len(tickers_list) > 1:
            if len(price_data.columns) < len(tickers_list):
                und_errors = np.setdiff1d(tickers_list, price_data.columns)
                gu.message('There was a problem loading data for the following underlyings:\n' + '\n'.join(und_errors))
                for und_error in und_errors:
                    if und_error in dependent_var_list:
                        dependent_var_list.remove(und_error)
                    if und_error in independent_var_list:
                        independent_var_list.remove(und_error)

    return dependent_var_list, independent_var_list
