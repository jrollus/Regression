import gui as gu
from datetime import date, timedelta

# Run the demo (if invoked from the command line):
if __name__ == '__main__':
    input_parameters = gu.InputParameter(time_window=120, minimum_r_squared=0.70, confidence_level=0.05,
                                         reg_model='OLS', data_source='Bloomberg',
                                         date_start=date.today() - timedelta(days=3*365),
                                         date_end=date.today() - timedelta(days=1))
    input_parameters.configure_traits()
