import traits.api as trapi
import traitsui.api as trui
import data_retrieval as dt
import calculations as cc

# Constants
nbr_max_time_windows = 5
screen_height = 400
screen_width = 600


class Message(trapi.HasPrivateTraits):
    """A class to show a message to the user"""
    message = trapi.Str


def message(message_input="", parent=None):
    """Displays a message to the user as a model window"""
    msg = Message(message=message_input)
    ui = msg.edit_traits(
        parent=parent,
        view=trui.View(
            ["message~", "|<>"], title='Error', buttons=["OK"], kind="modal", resizable=True,
            icon='reg.png', image='reg.png'
        ),
    )
    return ui.result


class InputParameter(trapi.HasTraits):
    """Class is used to input all of the user parameters through a guy"""
    dependent_var = trapi.Str
    independent_var = trapi.Str
    time_window = trapi.Int
    minimum_r_squared = trapi.Float
    confidence_level = trapi.Float
    reg_model = trapi.Enum("OLS", "NNLS")
    data_source = trapi.Enum("Bloomberg", "Telemaco")
    date_start = trapi.Date
    date_end = trapi.Date
    get_data_button = trapi.Button
    save_file = trapi.File
    v = trui.View(trui.HGroup(
                            trui.Item(name='dependent_var', style='custom', label='Dependent Variables'),
                            trui.Item(name='independent_var', style='custom', label='Independent Variables'),
                            trui.VGroup(trui.Item(name='date_start', label='Date Start'),
                                        trui.Item(name='date_end', label='Date End'),
                                        trui.Item(name='time_window', label='Time Window'),
                                        trui.Item(name='minimum_r_squared', label='Min. R Squared'),
                                        trui.Item(name='confidence_level', label='Conf. Level'),
                                        trui.Item(name='reg_model', label='Model'),
                                        trui.Item(name='data_source', label='Date Source'),
                                        trui.Item(name='save_file', label='Output Path'),
                                        trui.Item(name='get_data_button', label='Run Regression Analysis',
                                                  show_label=False),
                                        ),
                            show_border=True, label='Input Data'),
                resizable=True, title='Regression Tool', height=screen_height, width=screen_width,
                icon='reg.png', image='reg.png')

    def _get_data_button_fired(self):
        """Method to conduct the regression analysis and output the results to a csv file"""
        # Check whether an output path was selected
        if not self.save_file:
            message('You must select an output path before conducting the analysis.')
            return
        else:
            if self.save_file[-4:] !='.csv':
                self.save_file += '.csv'

        # Format input data from GUI
        dependent_var_list = self.dependent_var.strip().split('\n')
        independent_var_list = self.independent_var.strip().split('\n')

        # Retrieve price data
        price_data = dt.get_relevant_data(dependent_var_list + independent_var_list, self.date_start,
                                          self.date_end, self.data_source)
        dependent_var_list, independent_var_list = dt.check_data_retrieval_error(price_data, dependent_var_list,
                                                                                  independent_var_list,
                                                                                  self.data_source)
        # Generate all combinations of independent variables
        independent_var_combinations = cc.all_combinations(independent_var_list)

        # Check whether there was at least one dependent variable and one independent variable data properly loaded
        if (len(dependent_var_list) == 0) or (len(independent_var_list) == 0):
            message('You need at least one dependent and one independent variable to conduct the analysis.')

        # Compute log returns
        log_returns = cc.get_log_returns(price_data)

        # Compute realized volatilities
        realized_vols = cc.get_realized_vol(log_returns, self.time_window)

        # Run regression analysis
        regression_results = cc.get_regression_results(dependent_var_list, independent_var_combinations,
                                                       realized_vols, self.reg_model)

        # Process regression results
        regression_results = cc.process_regression_results(regression_results, self.minimum_r_squared,
                                                           self.confidence_level)
        # Output results to .CSV
        regression_results.to_csv(self.save_file)

