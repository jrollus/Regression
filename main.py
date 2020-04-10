import data_retrieval as dt
import calculations as cc
from datetime import date, timedelta

# Run the demo (if invoked from the command line):
if __name__ == '__main__':
    dependent_var = ['PIMEEHA ID Equity', 'PINEEHA ID Equity']
    independent_var = ['SX5E Index', 'SPX Index', 'HSI Index', 'JB1 Comdty', 'OE1 Comdty']
    # independent_var =['SX5E Index', 'SPX Index', 'HSI Index', 'JB1 Comdty', 'OE1 Comdty', 'RX1 Comdty', 'US1 Comdty',
    #                   'HYG US Equity', 'EWZ US Equity', 'EEM US Equity', 'XLE US Equity', 'XLF US Equity',
    #                   'KOSPI2 Index']
    independent_var_combinations = cc.all_combinations(independent_var)
    date_start = date.today() - timedelta(days=3 * 365)
    date_end = date.today() - timedelta(days=1)

    price_data = dt.get_relevant_data(dependent_var + independent_var, date_start, date_end, 'Bloomberg')
    tickers_list = dt.check_data_retrieval_error(price_data, dependent_var + independent_var)
    if len(tickers_list) <= 1 or tickers_list.count(dependent_var[0]) == 0:
        a = 1

    log_returns = cc.get_log_returns(price_data)
    realized_vols = cc.get_realized_vol(log_returns, 120)
    regression_results = cc.get_regression_results(dependent_var, independent_var_combinations, realized_vols)
    regression_results.to_csv('test.csv')