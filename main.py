import gui as gu
from datetime import date, timedelta

# Run the code (if invoked from the command line):
if __name__ == '__main__':
    input_parameters = gu.InputParameter(time_windows=[[120], [180], [260]], minimum_r_squared=0.70, adf_activate=True,
                                         confidence_level=0.05, reg_model='NNLS', data_source='Bloomberg',
                                         data_fill='Drop',
                                         date_start=date.today() - timedelta(days=3*365),
                                         date_cutoff=date.today() - timedelta(days=1 * 365),
                                         date_end=date.today() - timedelta(days=1))
    input_parameters.configure_traits()
