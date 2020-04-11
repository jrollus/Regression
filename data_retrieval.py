import pandas as pd
import numpy as np
import pandas_datareader.data as web
from pandas_datareader._utils import RemoteDataError
from xbbg import blp


def get_relevant_data(tickers_list, date_start, date_end, data_source):
    """Get financial data from Yahoo Finance for a given set of underlying and type of data"""
    if data_source == 'Yahoo':
        try:
            raw_data = web.DataReader(name=tickers_list, data_source='yahoo', start=date_start, end=date_end)
            selected_data = raw_data['Adj Close']
        except RemoteDataError:
            selected_data = pd.DataFrame(np.nan, index=[0], columns=tickers_list)

    elif data_source == 'Bloomberg':
        selected_data = blp.bdh(tickers=tickers_list, flds=['PX_LAST'],
                                start_date=date_start, end_date=date_end, adjust='all')

        if not selected_data.empty:
            selected_data.columns = selected_data.columns.droplevel(1)

    return selected_data


def check_data_retrieval_error(price_data, tickers_list):
    """Check whether there was an error retrieving data"""
    # Check whether data was retrieved successfully:
    data_source = 'Bloomberg'
    if data_source == 'Yahoo':
        if len(tickers_list) > 1:
            empty_col = []
            for column_name in price_data.columns:
                if price_data[column_name].isna().all():
                    empty_col.append(column_name)
                    price_data[column_name].drop
            if empty_col:
                message('There was a problem loading data for the following underlyings:\n' + '\n'.join(empty_col))
                return [x for x in tickers_list if x not in empty_col]
            else:
                return tickers_list
        else:
            return tickers_list

    elif data_source == 'Bloomberg':
        if len(tickers_list) > 1:
            if len(price_data.columns) < len(tickers_list):
                und_errors = np.setdiff1d(tickers_list, price_data.columns)
                message('There was a problem loading data for the following underlyings:\n' + '\n'.join(und_errors))
                return [x for x in tickers_list if x not in und_errors]
            else:
                return tickers_list
        else:
            return tickers_list