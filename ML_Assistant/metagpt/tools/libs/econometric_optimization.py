from __future__ import annotations
from metagpt.tools.tool_registry import register_tool

# Import basic packages
from itertools import combinations
import math
import pandas as pd

# Import econometric algorithms
import sys
sys.path.append("/home/kurtluo/GPT/ChatInterpreter/ML_Assistant/metagpt/tools/libs")
from econometric_algorithm import ordinary_least_square_regression, Panel_Data_OLS_regression, propensity_score_construction, propensity_score_regression, IV_2SLS_regression

#%%

class GeneralOptimizationMethods:
    """
    Optimize the target econometric analysis result in the given parametric space of available optimization aspects and methods
    """

    def __init__(self):
        pass
    
    def process_a_data_series(self, data_series, method_input):
        if method_input == "original":
            return data_series
        elif method_input == "log":
            adjusted_data_series = data_series.map(lambda x: math.log(x))
            adjusted_data_series.name = "log_" + data_series.name
            return adjusted_data_series
        elif method_input == "standardization":
            mean, std = data_series.mean(), data_series.std()
            adjusted_data_series = data_series.map(lambda x: (x - mean) / std)
            adjusted_data_series.name = "standardized_" + data_series.name
            return adjusted_data_series
        elif method_input == "mid_value_dummy":
            mid_value = data_series.quantile(0.5)
            adjusted_data_series = data_series.map(lambda x: 1 if x > mid_value else 0)
            adjusted_data_series.name = "mid_value_dummy_" + data_series.name
            return adjusted_data_series
        elif method_input == "remove_extreme":
            bottom_quantile, top_quantile = data_series.quantile(0.03), data_series.quantile(0.97)
            adjusted_data_series = data_series[(data_series >= bottom_quantile) & (data_series <= top_quantile)]
            adjusted_data_series.name = "remove_extreme_" + data_series.name
            return adjusted_data_series
        elif method_input == "remove_extreme_standardization":
            bottom_quantile, top_quantile = data_series.quantile(0.03), data_series.quantile(0.97)
            adjusted_data_series = data_series[(data_series >= bottom_quantile) & (data_series <= top_quantile)]
            mean, std = adjusted_data_series.mean(), adjusted_data_series.std()
            adjusted_data_series = adjusted_data_series.map(lambda x: (x - mean) / std)
            adjusted_data_series.name = "remove_extreme_standardization_" + data_series.name
            return adjusted_data_series
        
    def dictlist_listdict_transformation(self, original_dictlist):
        result_list = []
        result_list_raw = []
        for each_index in original_dictlist:
            for each_value in original_dictlist[each_index]:
                result_list_raw.append(each_index + "__________" + each_value)
        all_candidate_combinations = combinations(result_list_raw, len(original_dictlist))
        for each_result in all_candidate_combinations:
            temp_combination_result = {}
            fail = False
            for each_value in list(each_result):
                key, value = each_value.split("__________")[0], each_value.split("__________")[1]
                if key not in temp_combination_result:
                    temp_combination_result[key] = value
                else:
                    fail = True
                    break
            if fail is False:
                result_list.append(temp_combination_result)                
        return result_list
        
#%%

@register_tool(tags=["econometric optimization"])
class General_OLS_Optimization(GeneralOptimizationMethods):
    """
    Optimize the general OLS regression result in the given optimization parametric space.
    """
    
    def __init__(self, 
                 dependent_variable: pd.Series, 
                 treatment_variable: pd.Series, 
                 treatment_is_dummy: bool, 
                 control_variables = None, 
                 covariance_requirement: bool = False, 
                 categorial_control_name: list = [], 
                 restriction_dict: dict = {}, 
                 **params):
        
        """
        Initialize self and generate the optimization plan. The input data Series or DataFrames should not contain any nan value.
        The optimization space includes "dependent_variable_transformation", "treatment_variable_transformation", "control_variable_selection", 
        "control_variable_transformation", and finally "covariance_requirement". 
        WHEN CALLING THIS OBJECT, BE SURE USER's RESTRICTIONS (IF ANY) ARE CAREFULLY CONTROLLED IN THE ARGUMENT 'restriction_dict'. 
        If any aspect is not allowed by the user, put the aspect name into the argument 'restriction_dict' as the key and bool False as the value.
        After successfully calling the attribute function "optimization", you can obtain the best optimization setting via the attribute variable "self.best_setting".

        Args:
            dependent_variable (pd.Series): Target dependent variable, which now is required to be a continuous numerical variable.
            treatment_variable (pd.Series): Target treatment variable, which should be a dummy or continuous variable.
            treatment_is_dummy (bool): Denote if treatment variable is not continuous but a CATEGORICAL OR DUMMY variable. 
            control_variables (pd.DataFrame or None, optional): A dataframe of control variables, which should not contain the intercept. If there is no specified control variable, leave it the default None value. 
            covariance_requirement (bool, optional): Specify if the OLS regression method requires COVARIANCE adjustment. If required, this aspect will also be included for optimization. Default is False.
            categorial_control_name (list, optional): A list of control variable names, each of which in this list indicate the denoted control variable is not continuous but a categorial variable.
            restriction_dict (dict, optional): Detailed restrictions towards the optimization methods. For the given optimization space, if there is restriction towards any method, add the method in this input with the name as the key and bool False as the value. Should leave empty if no speficied restriction needed.
            **params (optional): Should contain all other information for the specified econometric analysis method.
        """
        
        # Initialize the attributes
        self.dependent_variable = dependent_variable
        self.treatment_variable = treatment_variable
        self.treatment_is_dummy = treatment_is_dummy
        self.covariance_requirement = covariance_requirement
        self.control_variables = control_variables
        self.categorial_control_name = categorial_control_name
        self.params_dict = params
        
        # Categorical variables cannot be directly figured out, but at least dummy variables can
        if self.control_variables is not None:
            for each_control in self.control_variables.columns:
                if self.control_variables[each_control].unique().shape[0] == 2 and each_control not in self.categorial_control_name:
                    print("Control Variable:", each_control, "is dummy but not included in categorical control name list. Has been added now!")
                    self.categorial_control_name.append(each_control)
        
        # Set the optimization plan
        optimization_plan = {
            "dependent_variable_transformation": True, 
            "treatment_variable_transformation": True, 
            "control_variable_selection": True, 
            "control_variable_transformation": True, 
            "covariance_requirement": covariance_requirement
            }
        
        # Check if any optimization method cannot be implemented
        if self.treatment_is_dummy is True:
            optimization_plan["treatment_variable_transformation"] = False
        if self.control_variables is None:
            optimization_plan["control_variable_selection"] = False
            optimization_plan["control_variable_transformation"] = False
        
        # If user specifies some particular methods not to implement, adjust the plan, and then finalize the plan
        for each_restriction in restriction_dict:
            if each_restriction in optimization_plan:
                optimization_plan[each_restriction] = False
            else:
                raise RuntimeError("Specified restriction does not direct to an optimization method. Please check!")
        self.plan = optimization_plan
        
        # Store the best setting after optimization
        self.best_setting = None
        
    # -------------------------------------------------------------------------    
        
    def optimization(self, target_type: str = "neg_pvalue"):
        
        """
        Generate the optimization parametric space and optimize the final analysis result.
        Also provide optimization parametric sets for top 10 results.
        
        Args:
            target_type (str, optional): An instruction about the way to evaluate the regression outcome. When set default as "neg_pvalue", the evaluation criteria is the negative value of treatment variable's coefficient p-value. Can also choose "rsquared" which output the adjusted R-squared value for the regression.
        """
        
        # Check the evaluation target input
        if target_type not in ["neg_pvalue", "rsquared"]:
            raise RuntimeError("Evaluation Target Type Not Supported, Please Check!")
        
        # Generate detailed parametric sets based on the plan
        detailed_parametric_space = {}
        
        # Check towards dependent variable transformation
        if self.plan["dependent_variable_transformation"] is True:
            if self.dependent_variable[self.dependent_variable <= 0].shape[0] == 0:
                detailed_parametric_space["dependent_variable_transformation"] = ["original", "log", "remove_extreme"]
            else:
                detailed_parametric_space["dependent_variable_transformation"] = ["original", "remove_extreme"]
        else:
            detailed_parametric_space["dependent_variable_transformation"] = ["original"]
        
        # Check towards treatment variable transformation
        if self.plan["treatment_variable_transformation"] is True:
            if self.treatment_variable[self.treatment_variable <= 0].shape[0] == 0:
                detailed_parametric_space["treatment_variable_transformation"] = ["original", "log", "remove_extreme", "standardization", "remove_extreme_standardization"]
            else:
                detailed_parametric_space["treatment_variable_transformation"] = ["original", "remove_extreme", "standardization", "remove_extreme_standardization"]
        else:
            detailed_parametric_space["treatment_variable_transformation"] = ["original"]

        # Check towards control variable selection
        if self.plan["control_variable_selection"] is True:
            all_control_variable_amount = self.control_variables.shape[1]
            all_control_variable_list = list(self.control_variables.columns)
            all_possible_control_variable_combinations = []
            all_possible_control_variable_combinations.append([])
            for i in range(1, all_control_variable_amount + 1):
                for ii in combinations(all_control_variable_list, i):
                    all_possible_control_variable_combinations.append(list(ii))
            detailed_parametric_space["control_variable_selection"] = all_possible_control_variable_combinations            
        else:
            if self.control_variables is not None:
                detailed_parametric_space["control_variable_selection"] = [list(self.control_variables.columns)]
            else:
                detailed_parametric_space["control_variable_selection"] = [[]]
                
        # Check towards control variable transformation
        if self.plan["control_variable_transformation"] is True:
            control_variable_transformation_dict = {}
            for each_control_variable in self.control_variables.columns:
                if each_control_variable in self.categorial_control_name:
                    control_variable_transformation_dict[each_control_variable] = ["original"]
                else:
                    if self.control_variables[each_control_variable][self.control_variables[each_control_variable] <= 0].shape[0] == 0:
                        control_variable_transformation_dict[each_control_variable] = ["original", "log", "standardization", "mid_value_dummy"]
                    else:
                        control_variable_transformation_dict[each_control_variable] = ["original", "standardization", "mid_value_dummy"]
            detailed_parametric_space["control_variable_transformation"] = control_variable_transformation_dict
        else:
            control_variable_transformation_dict = {}
            if self.control_variables is not None:
                for each_control_variable in self.control_variables.columns:
                    control_variable_transformation_dict[each_control_variable] = ["original"]
            detailed_parametric_space["control_variable_transformation"] = control_variable_transformation_dict

        # Check towards cov requirement
        if self.plan["covariance_requirement"] is True:
            detailed_parametric_space["covariance_requirement"] = ["HC0", "HC1", "HC2", "HC3"]    
        else:
            detailed_parametric_space["covariance_requirement"] = [None]
            
        # ---------------------------------------------------------------------
        
        # Adjust the transformation data type (from a dict of list of str to a list of dict)
        detailed_parametric_space["control_variable_transformation"] = self.dictlist_listdict_transformation(detailed_parametric_space["control_variable_transformation"])
        
        # Record all results
        all_setting_dict_record_list = []
        all_setting_dict_record = {}
        count = 0
        for each_dependent_variable_transformation in detailed_parametric_space["dependent_variable_transformation"]:
            for each_treatment_variable_transformation in detailed_parametric_space["treatment_variable_transformation"]:
                for each_control_variable_selection in detailed_parametric_space["control_variable_selection"]:
                    for each_control_variable_transformation in detailed_parametric_space["control_variable_transformation"]:
                        for each_covariance_requirement in detailed_parametric_space["covariance_requirement"]:
                            each_control_variable_transformation = {i: each_control_variable_transformation[i] for i in each_control_variable_selection}
                            temp_dict = {
                                "dependent_variable_transformation": each_dependent_variable_transformation, 
                                "treatment_variable_transformation": each_treatment_variable_transformation, 
                                "control_variable_selection": each_control_variable_selection, 
                                "control_variable_transformation": each_control_variable_transformation, 
                                "covariance_requirement": each_covariance_requirement
                                }
                            if temp_dict not in all_setting_dict_record_list:
                                count += 1
                                all_setting_dict_record[count] = temp_dict
                                all_setting_dict_record_list.append(temp_dict)
        all_results_series = pd.Series(index = all_setting_dict_record.keys())
        print("Total Optimization Trials:", all_results_series.shape[0])
            
        # ---------------------------------------------------------------------
        
        # Go through each setting, run the regression and store the result
        print("Starting to Conduct Optimization! Will need some time...")
        for each_setting_count in all_setting_dict_record:
            
            # Process each setting treatment
            dependent_variable = self.process_a_data_series(self.dependent_variable, all_setting_dict_record[each_setting_count]["dependent_variable_transformation"])
            treatment_variable = self.process_a_data_series(self.treatment_variable, all_setting_dict_record[each_setting_count]["treatment_variable_transformation"])
            if self.control_variables is not None:
                control_variables = self.control_variables[all_setting_dict_record[each_setting_count]["control_variable_selection"]]
                if control_variables.shape[1] == 0:
                    control_variables = None
                else:
                    for each_control in control_variables:
                        control_variables.loc[:, each_control] = self.process_a_data_series(control_variables[each_control], all_setting_dict_record[each_setting_count]["control_variable_transformation"][each_control]).copy()
            else:
                control_variables = None
            cov_type = all_setting_dict_record[each_setting_count]["covariance_requirement"]
            
            # Since dependent variable and treatment variable might have remove extreme operation, should remember to match data after all transformations
            dependent_variable = dependent_variable[dependent_variable.index.isin(treatment_variable.index)]
            treatment_variable = treatment_variable.loc[dependent_variable.index]
            if control_variables is not None:
                control_variables = control_variables.loc[dependent_variable.index]
            
            # Run the regression, obtain and store the result
            final_result = ordinary_least_square_regression(dependent_variable, treatment_variable, control_variables, cov_type = cov_type, target_type = target_type, output_tables = False)
            all_results_series[each_setting_count] = final_result
    
        # Print out the top 10 result settings
        all_results_series = all_results_series.sort_values(ascending = False)
        if all_results_series.shape[0] < 10:
            selected_range = list(range(1, all_results_series.shape[0] + 1))
        else:
            selected_range = list(range(1, 10 + 1))
        print("==============================================================")
        print("The Best 10 Results and Settings:")
        print("--------------------------------------------------------------")
        for i in selected_range:
            selected_index = all_results_series.index[i - 1]
            print("--------------------------------------------------------------")
            print("Count:", i)
            print("Result:", all_results_series.iloc[i - 1])
            print("Setting Information:", all_setting_dict_record[selected_index])
            print("--------------------------------------------------------------")
        print("==============================================================")

        # Output the best result
        selected_setting_count = all_results_series.index[0]
        
        # Process each setting treatment
        dependent_variable = self.process_a_data_series(self.dependent_variable, all_setting_dict_record[selected_setting_count]["dependent_variable_transformation"])
        treatment_variable = self.process_a_data_series(self.treatment_variable, all_setting_dict_record[selected_setting_count]["treatment_variable_transformation"])
        if self.control_variables is not None:
            control_variables = self.control_variables[all_setting_dict_record[selected_setting_count]["control_variable_selection"]]
            if control_variables.shape[1] == 0:
                control_variables = None
            else:
                for each_control in control_variables:
                    control_variables.loc[:, each_control] = self.process_a_data_series(control_variables[each_control], all_setting_dict_record[selected_setting_count]["control_variable_transformation"][each_control]).copy()
        else:
            control_variables = None
        cov_type = all_setting_dict_record[selected_setting_count]["covariance_requirement"]
            
        # Since dependent variable and treatment variable might have remove extreme operation, should remember to match data after all transformations
        dependent_variable = dependent_variable[dependent_variable.index.isin(treatment_variable.index)]
        treatment_variable = treatment_variable.loc[dependent_variable.index]
        if control_variables is not None:
            control_variables = control_variables.loc[dependent_variable.index]
        
        # Run the regression
        final_result = ordinary_least_square_regression(dependent_variable, treatment_variable, control_variables, cov_type = cov_type, target_type = None, output_tables = True)
        
        # Store the best setting
        self.best_setting = all_setting_dict_record[selected_setting_count]
   
#%%
    
@register_tool(tags=["econometric optimization"])
class Panel_OLS_Optimization(GeneralOptimizationMethods):
    """
    Optimize the Panel Data OLS regression result in the given optimization parametric space.
    """
    
    def __init__(self, 
                 dependent_variable: pd.Series, 
                 treatment_variable: pd.Series, 
                 treatment_is_dummy: bool, 
                 control_variables = None, 
                 categorial_control_name: list = [], 
                 entity_effect_specification = False, 
                 time_effect_specification = False, 
                 other_effect_specification = None, 
                 cov_type_specification = None, 
                 restriction_dict: dict = {}, 
                 **params):
        
        """
        Initialize self and generate the optimization plan. The input data Series or DataFrames should not contain any nan value.
        The optimization space includes "dependent_variable_transformation", "treatment_variable_transformation", "control_variable_selection", 
        "control_variable_transformation", "entity_effect_selection", "time_effect_selection", "other_effect_selection", and finally "covariance_requirement". 
        WHEN CALLING THIS OBJECT, BE SURE USER's RESTRICTIONS (IF ANY) ARE CAREFULLY CONTROLLED IN THE ARGUMENT 'restriction_dict'. 
        If any aspect is not allowed by the user, put the aspect name into the argument 'restriction_dict' as the key and bool False as the value.
        After successfully calling the attribute function "optimization", you can obtain the best optimization setting via the attribute variable "self.best_setting".

        Args:
            dependent_variable (pd.Series): Target dependent variable, which now is required to be a continuous numerical variable. The index of the series should be entity-time multi-index.
            treatment_variable (pd.Series): Target treatment variable, which should be a dummy or continuous variable. The index of the series should be entity-time multi-index.
            treatment_is_dummy (bool): Denote if treatment variable is not continuous but a CATEGORICAL OR DUMMY variable. 
            control_variables (pd.DataFrame or None, optional): A dataframe of control variables, which should not contain the intercept. The index of the series should be entity-time multi-index. If there is no specified control variable, leave it the default None value.
            categorial_control_name (list, optional): A list of control variable names, each of which in this list indicate the denoted control variable is not continuous but a categorial variable.
            entity_effect_specification (bool, optional): Denote if entity effect is specified to be included in the PanelOLS optimization algorithm. However, if specified as True, the "entity_effect_selection" optimization aspect will automatically be set False to ignore entity effect selection; if specified as False, the "entity_effect_selection" optimization will include entity effect selection, unless user specifies "entity_effect_selection" = False in the input "restriction_dict".
            time_effect_specification (bool, optional): Denote if time effect is specified to be included in the PanelOLS optimization algorithm. However, if specified as True, the "time_effect_selection" optimization aspect will automatically be set False to ignore time effect selection; if specified as False, the "time_effect_selection" optimization will include time effect selection, unless user specifies "time_effect_selection" = False in the input "restriction_dict".
            other_effect_specification (pd.DataFrame or None, optional): Denote if there is any other effect to be included in the PanelOLS optimization algorithm. If the input is a pd.DataFrame, the index should be entity-time multi-index, the input data MUST BE CATEGORICAL, and the "other_effect_selection" optimization will include other effect selection, unless user specifies "other_effect_selection" = False in the input "restriction_dict"; however, if the input is None, the "other_effect_selection" optimization aspect will automatically be set False to ignore other effect selection.
            cov_type_specification (str or None, optional): Specify if the PanelOLS regression method requires COVARIANCE adjustment. Five covariance estimators are supported: "unadjusted" for homoskedastic residual, "robust" for heteroskedasticity control, "cluster_entity" for entity clustering, "cluster_time" for time clustering, and "cluster_both" for entity-time two-way clustering. However, if specified, the "covariance_requirement" optimization aspect will automatically be set False. To allow optimization towards covariance requirement, leave this input to be its default value None. Also, this input should be left None if no covariance requirement optimization is allowed.
            restriction_dict (dict, optional): Detailed restrictions towards the optimization methods. For the given optimization space, if there is restriction towards any method, add the method in this input with the name as the key and bool False as the value. Should leave empty if no speficied restriction needed.
            **params (optional): Should contain all other information for the specified econometric analysis method.
        """
        
        # Initialize the attributes
        self.dependent_variable = dependent_variable
        self.treatment_variable = treatment_variable
        self.treatment_is_dummy = treatment_is_dummy
        self.control_variables = control_variables
        self.categorial_control_name = categorial_control_name
        self.entity_effect_specification = entity_effect_specification
        self.time_effect_specification = time_effect_specification
        self.other_effect_specification = other_effect_specification
        self.cov_type_specification = cov_type_specification
        self.params_dict = params

        # Categorical variables cannot be directly figured out, but at least dummy variables can
        if self.control_variables is not None:
            for each_control in self.control_variables.columns:
                if self.control_variables[each_control].unique().shape[0] == 2 and each_control not in self.categorial_control_name:
                    print("Control Variable:", each_control, "is dummy but not included in categorical control name list. Has been added now!")
                    self.categorial_control_name.append(each_control)
        
        # Set the optimization plan
        optimization_plan = {
            "dependent_variable_transformation": True, 
            "treatment_variable_transformation": True, 
            "control_variable_selection": True, 
            "control_variable_transformation": True, 
            "entity_effect_selection": True, 
            "time_effect_selection": True, 
            "other_effect_selection": True, 
            "covariance_requirement": True
            }
        
        # Check if any optimization method cannot be implemented
        if self.treatment_is_dummy is True:
            optimization_plan["treatment_variable_transformation"] = False
        if self.control_variables is None:
            optimization_plan["control_variable_selection"] = False
            optimization_plan["control_variable_transformation"] = False
        if entity_effect_specification is True:
            optimization_plan["entity_effect_selection"] = False
        if time_effect_specification is True:
            optimization_plan["time_effect_selection"] = False
        if other_effect_specification is None:
            optimization_plan["other_effect_selection"] = False
        if cov_type_specification is not None:
            optimization_plan["covariance_requirement"] = False
        
        # If user specifies some particular methods not to implement, adjust the plan, and then finalize the plan
        for each_restriction in restriction_dict:
            if each_restriction in optimization_plan:
                optimization_plan[each_restriction] = False
            else:
                raise RuntimeError("Specified restriction does not direct to an optimization method. Please check!")
        self.plan = optimization_plan
        
        # Check if inputs are proper formatted
        if cov_type_specification is not None and cov_type_specification not in ["unadjusted", "robust", "cluster_entity", "cluster_time", "cluster_both"]:
            raise RuntimeError("Covariance type input unsupported! If specified, this input supports 'unadjusted', 'robust', 'cluster_entity', 'cluster_time' and 'cluster_both' as possible inputs!")
        currently_required_count_effects = 0
        if entity_effect_specification is True:
            currently_required_count_effects += 1
        if time_effect_specification is True:
            currently_required_count_effects += 1
        if other_effect_specification is not None and optimization_plan["other_effect_selection"] is False:
            currently_required_count_effects += other_effect_specification.shape[1]
            if currently_required_count_effects > 2:
                raise RuntimeError("At most two effects allowed! Please note that now there have already been " + str(currently_required_count_effects) + " effects required in total!")
        elif other_effect_specification is not None and optimization_plan["other_effect_selection"] is True:
            if currently_required_count_effects == 2:
                print("Warning: Entity and Time Effects are already required by user, such that other effect(s) cannot be further included! Will ignore the other effect input!")
                self.other_effect_specification = None
                self.plan["other_effect_selection"] = False
        
        # Store the best setting after optimization
        self.best_setting = None
        
    # -------------------------------------------------------------------------    
        
    def optimization(self, target_type: str = "neg_pvalue"):
        
        """
        Generate the optimization parametric space and optimize the final analysis result.
        Also provide optimization parametric sets for top 10 results.
        
        Args:
            target_type (str, optional): An instruction about the way to evaluate the regression outcome. When set default as "neg_pvalue", the evaluation criteria is the negative value of treatment variable's coefficient p-value. Can also choose "rsquared" which output the adjusted R-squared value for the regression.
        """
        
        # Check the evaluation target input
        if target_type not in ["neg_pvalue", "rsquared"]:
            raise RuntimeError("Evaluation Target Type Not Supported, Please Check!")
        
        # Generate detailed parametric sets based on the plan
        detailed_parametric_space = {}
        
        # Check towards dependent variable transformation
        if self.plan["dependent_variable_transformation"] is True:
            if self.dependent_variable[self.dependent_variable <= 0].shape[0] == 0:
                detailed_parametric_space["dependent_variable_transformation"] = ["original", "log", "remove_extreme"]
            else:
                detailed_parametric_space["dependent_variable_transformation"] = ["original", "remove_extreme"]
        else:
            detailed_parametric_space["dependent_variable_transformation"] = ["original"]
        
        # Check towards treatment variable transformation
        if self.plan["treatment_variable_transformation"] is True:
            if self.treatment_variable[self.treatment_variable <= 0].shape[0] == 0:
                detailed_parametric_space["treatment_variable_transformation"] = ["original", "log", "remove_extreme", "standardization", "remove_extreme_standardization"]
            else:
                detailed_parametric_space["treatment_variable_transformation"] = ["original", "remove_extreme", "standardization", "remove_extreme_standardization"]
        else:
            detailed_parametric_space["treatment_variable_transformation"] = ["original"]

        # Check towards control variable selection
        if self.plan["control_variable_selection"] is True:
            all_control_variable_amount = self.control_variables.shape[1]
            all_control_variable_list = list(self.control_variables.columns)
            all_possible_control_variable_combinations = []
            all_possible_control_variable_combinations.append([])
            for i in range(1, all_control_variable_amount + 1):
                for ii in combinations(all_control_variable_list, i):
                    all_possible_control_variable_combinations.append(list(ii))
            detailed_parametric_space["control_variable_selection"] = all_possible_control_variable_combinations            
        else:
            if self.control_variables is not None:
                detailed_parametric_space["control_variable_selection"] = [list(self.control_variables.columns)]
            else:
                detailed_parametric_space["control_variable_selection"] = [[]]
            
        # Check towards control variable transformation
        if self.plan["control_variable_transformation"] is True:
            control_variable_transformation_dict = {}
            for each_control_variable in self.control_variables.columns:
                if each_control_variable in self.categorial_control_name:
                    control_variable_transformation_dict[each_control_variable] = ["original"]
                else:
                    if self.control_variables[each_control_variable][self.control_variables[each_control_variable] <= 0].shape[0] == 0:
                        control_variable_transformation_dict[each_control_variable] = ["original", "log", "standardization", "mid_value_dummy"]
                    else:
                        control_variable_transformation_dict[each_control_variable] = ["original", "standardization", "mid_value_dummy"]
            detailed_parametric_space["control_variable_transformation"] = control_variable_transformation_dict
        else:
            control_variable_transformation_dict = {}
            if self.control_variables is not None:
                for each_control_variable in self.control_variables.columns:
                    control_variable_transformation_dict[each_control_variable] = ["original"]
            detailed_parametric_space["control_variable_transformation"] = control_variable_transformation_dict

        # Check towards all effect selection
        detailed_parametric_space["all_effect_selection"] = []
        currently_required_count_effects = 0
        if self.entity_effect_specification is True:
            currently_required_count_effects += 1
        if self.time_effect_specification is True:
            currently_required_count_effects += 1
        if self.other_effect_specification is not None and self.plan["other_effect_selection"] is False:
            currently_required_count_effects += self.other_effect_specification.shape[1]
        
        # If currently already two effects, no need to select any effect then
        if currently_required_count_effects == 2:
            detailed_parametric_space["all_effect_selection"].append({
                "entity_effect": self.entity_effect_specification, 
                "time_effect": self.time_effect_specification, 
                "other_effect": self.other_effect_specification
                })
        
        # If there is only 0 or 1 effect required, can select other effect(s) to be included then
        else:
            
            # This version does not select any extra effect
            if self.plan["other_effect_selection"] is True:
                detailed_parametric_space["all_effect_selection"].append({
                    "entity_effect": self.entity_effect_specification, 
                    "time_effect": self.time_effect_specification, 
                    "other_effect": None
                    })
            else:
                detailed_parametric_space["all_effect_selection"].append({
                    "entity_effect": self.entity_effect_specification, 
                    "time_effect": self.time_effect_specification, 
                    "other_effect": self.other_effect_specification
                    })
            
            # This version draws combinations for effect selection
            remaining_allowed_effect_count = 2 - currently_required_count_effects
            possible_effect_list = []
            if self.plan["entity_effect_selection"] is True:
                possible_effect_list.append("entity")
            if self.plan["time_effect_selection"] is True:
                possible_effect_list.append("time")
            if self.plan["other_effect_selection"] is True:
                possible_effect_list += list(self.other_effect_specification.columns)
            possible_effect_combinations = []
            for i in range(1, remaining_allowed_effect_count + 1):
                for ii in combinations(possible_effect_list, i):
                    possible_effect_combinations.append(list(ii))
            for each_result in possible_effect_combinations:
                candidate_effect_selection_dict = {
                    "entity_effect": None, 
                    "time_effect": None, 
                    "other_effect": None
                    }
                if "entity" in each_result:
                    candidate_effect_selection_dict["entity_effect"] = True
                else:
                    candidate_effect_selection_dict["entity_effect"] = self.entity_effect_specification
                if "time" in each_result:
                    candidate_effect_selection_dict["time_effect"] = True
                else:
                    candidate_effect_selection_dict["time_effect"] = self.time_effect_specification
                selected_other_effect_list = []
                for each_other_effect_column in self.other_effect_specification.columns:
                    if each_other_effect_column in each_result:
                        selected_other_effect_list.append(each_other_effect_column)
                if len(selected_other_effect_list) != 0:
                    candidate_effect_selection_dict["other_effect"] = self.other_effect_specification[selected_other_effect_list]
                elif self.plan["other_effect_selection"] is False:
                    candidate_effect_selection_dict["other_effect"] = self.other_effect_specification
            possible_effect_list.append(candidate_effect_selection_dict)
        detailed_parametric_space["all_effect_selection"] += possible_effect_list

        # Check towards cov requirement
        if self.plan["covariance_requirement"] is True:
            detailed_parametric_space["covariance_requirement"] = ["unadjusted", "robust", "cluster_entity", "cluster_time", "cluster_both"]
        else:
            detailed_parametric_space["covariance_requirement"] = [self.cov_type_specification]
            
        # ---------------------------------------------------------------------
        
        # Adjust the transformation data type (from a dict of list of str to a list of dict)
        detailed_parametric_space["control_variable_transformation"] = self.dictlist_listdict_transformation(detailed_parametric_space["control_variable_transformation"])
        
        # Record all results
        all_setting_dict_record_list = []
        all_setting_dict_record = {}
        count = 0
        for each_dependent_variable_transformation in detailed_parametric_space["dependent_variable_transformation"]:
            for each_treatment_variable_transformation in detailed_parametric_space["treatment_variable_transformation"]:
                for each_control_variable_selection in detailed_parametric_space["control_variable_selection"]:
                    for each_control_variable_transformation in detailed_parametric_space["control_variable_transformation"]:
                        for each_effect_selection in detailed_parametric_space["other_effect_selection"]:
                            for each_covariance_requirement in detailed_parametric_space["covariance_requirement"]:
                                each_control_variable_transformation = {i: each_control_variable_transformation[i] for i in each_control_variable_selection}
                                temp_dict = {
                                    "dependent_variable_transformation": each_dependent_variable_transformation, 
                                    "treatment_variable_transformation": each_treatment_variable_transformation, 
                                    "control_variable_selection": each_control_variable_selection, 
                                    "control_variable_transformation": each_control_variable_transformation, 
                                    "all_effect_selection": each_effect_selection,
                                    "covariance_requirement": each_covariance_requirement
                                    }
                                if temp_dict not in all_setting_dict_record_list:
                                    count += 1
                                    all_setting_dict_record[count] = temp_dict
                                    all_setting_dict_record_list.append(temp_dict)
        all_results_series = pd.Series(index = all_setting_dict_record.keys())
        print("Total Optimization Trials:", all_results_series.shape[0])
            
        # ---------------------------------------------------------------------
        
        # Go through each setting, run the regression and store the result
        print("Starting to Conduct Optimization! Will need some time...")
        for each_setting_count in all_setting_dict_record:
            
            # Process each setting treatment
            dependent_variable = self.process_a_data_series(self.dependent_variable, all_setting_dict_record[each_setting_count]["dependent_variable_transformation"])
            treatment_variable = self.process_a_data_series(self.treatment_variable, all_setting_dict_record[each_setting_count]["treatment_variable_transformation"])
            if self.control_variables is not None:
                control_variables = self.control_variables[all_setting_dict_record[each_setting_count]["control_variable_selection"]]
                if control_variables.shape[1] == 0:
                    control_variables = None
                else:
                    for each_control in control_variables:
                        control_variables.loc[:, each_control] = self.process_a_data_series(control_variables[each_control], all_setting_dict_record[each_setting_count]["control_variable_transformation"][each_control]).copy()
            else:
                control_variables = None
            entity_effect = all_setting_dict_record[each_setting_count]["all_effect_selection"]["entity_effect"]
            time_effect = all_setting_dict_record[each_setting_count]["all_effect_selection"]["time_effect"]
            other_effect = all_setting_dict_record[each_setting_count]["all_effect_selection"]["other_effect"]
            cov_type = all_setting_dict_record[each_setting_count]["covariance_requirement"]
            
            # Since dependent variable and treatment variable might have remove extreme operation, should remember to match data after all transformations
            dependent_variable = dependent_variable[dependent_variable.index.isin(treatment_variable.index)]
            treatment_variable = treatment_variable.loc[dependent_variable.index]
            if control_variables is not None:
                control_variables = control_variables.loc[dependent_variable.index]
            if other_effect is not None:
                other_effect = other_effect.loc[dependent_variable.index]
            
            # Run the regression, obtain and store the result
            final_result = Panel_Data_OLS_regression(dependent_variable, treatment_variable, control_variables, entity_effect = entity_effect, time_effect = time_effect, other_effect = other_effect, cov_type = cov_type, target_type = target_type, output_tables = False)
            all_results_series[each_setting_count] = final_result
    
        # Print out the top 10 result settings
        all_results_series = all_results_series.sort_values(ascending = False)
        if all_results_series.shape[0] < 10:
            selected_range = list(range(1, all_results_series.shape[0] + 1))
        else:
            selected_range = list(range(1, 10 + 1))
        print("==============================================================")
        print("The Best 10 Results and Settings:")
        print("--------------------------------------------------------------")
        for i in selected_range:
            selected_index = all_results_series.index[i - 1]
            print("--------------------------------------------------------------")
            print("Count:", i)
            print("Result:", all_results_series.iloc[i - 1])
            print("Setting Information:", all_setting_dict_record[selected_index])
            print("--------------------------------------------------------------")
        print("==============================================================")

        # Output the best result
        selected_setting_count = all_results_series.index[0]
        
        # Process each setting treatment
        dependent_variable = self.process_a_data_series(self.dependent_variable, all_setting_dict_record[selected_setting_count]["dependent_variable_transformation"])
        treatment_variable = self.process_a_data_series(self.treatment_variable, all_setting_dict_record[selected_setting_count]["treatment_variable_transformation"])
        
        if self.control_variables is not None:
            control_variables = self.control_variables[all_setting_dict_record[selected_setting_count]["control_variable_selection"]]
            if control_variables.shape[1] == 0:
                control_variables = None
            else:
                for each_control in control_variables:
                    control_variables.loc[:, each_control] = self.process_a_data_series(control_variables[each_control], all_setting_dict_record[selected_setting_count]["control_variable_transformation"][each_control]).copy()
        else:
            control_variables = None
        entity_effect = all_setting_dict_record[selected_setting_count]["all_effect_selection"]["entity_effect"]
        time_effect = all_setting_dict_record[selected_setting_count]["all_effect_selection"]["time_effect"]
        other_effect = all_setting_dict_record[selected_setting_count]["all_effect_selection"]["other_effect"]
        cov_type = all_setting_dict_record[selected_setting_count]["covariance_requirement"]
            
        # Since dependent variable and treatment variable might have remove extreme operation, should remember to match data after all transformations
        dependent_variable = dependent_variable[dependent_variable.index.isin(treatment_variable.index)]
        treatment_variable = treatment_variable.loc[dependent_variable.index]
        if control_variables is not None:
            control_variables = control_variables.loc[dependent_variable.index]
        if other_effect is not None:
            other_effect = other_effect.loc[dependent_variable.index]
            
        # Run the regression
        final_result = Panel_Data_OLS_regression(dependent_variable, treatment_variable, control_variables, entity_effect = entity_effect, time_effect = time_effect, other_effect = other_effect, cov_type = cov_type, target_type = None, output_tables = True)
        
        # Store the best setting
        self.best_setting = all_setting_dict_record[selected_setting_count]

#%%

@register_tool(tags=["econometric optimization"])
class PS_Regression_Optimization(GeneralOptimizationMethods):
    """
    Optimize the propensity score regression result in the given optimization parametric space.
    """
    
    def __init__(self, 
                 dependent_variable: pd.Series, 
                 treatment_variable: pd.Series, 
                 treatment_is_dummy: bool, 
                 control_variables: pd.DataFrame, 
                 covariance_requirement: bool = False, 
                 categorial_control_name: list = [], 
                 trimming_requirement = None, 
                 restriction_dict: dict = {}, 
                 **params):
        
        """
        Initialize self and generate the optimization plan. The input data Series or DataFrames should not contain any nan value.
        The optimization space includes "dependent_variable_transformation", "treatment_variable_transformation", "control_variable_selection", 
        "control_variable_transformation", "sample_trimming_test", and finally "covariance_requirement". 
        WHEN CALLING THIS OBJECT, BE SURE USER's RESTRICTIONS (IF ANY) ARE CAREFULLY CONTROLLED IN THE ARGUMENT 'restriction_dict'. 
        If any aspect is not allowed by the user, put the aspect name into the argument 'restriction_dict' as the key and bool False as the value.
        After successfully calling the attribute function "optimization", you can obtain the best optimization setting via the attribute variable "self.best_setting".

        Args:
            dependent_variable (pd.Series): Target dependent variable, which now is required to be a continuous numerical variable.
            treatment_variable (pd.Series): Target treatment variable, which should be a dummy or continuous variable.
            treatment_is_dummy (bool): Denote if treatment variable is not continuous but a CATEGORICAL OR DUMMY variable. 
            control_variables (pd.DataFrame): A dataframe of control variables, which should not contain the intercept. 
            covariance_requirement (bool, optional): Specify if the Propensity Score regression method requires COVARIANCE adjustment. If required, this aspect will also be included for optimization. Default is False.
            categorial_control_name (list, optional): A list of control variable names, each of which in this list indicate the denoted control variable is not continuous but a categorial variable.
            trimming_requirement (list or None): A list containing propensity-score-based sample trimming requirement, for example, [0.05, 0.95]. However, if specified, the "sample_trimming_test" optimization aspect will automatically be set False. To allow optimization towards sample trimming method, leave this input to be its default value None. Also, this input should be left None if no sample trimming is allowed.
            restriction_dict (dict, optional): Detailed restrictions towards the optimization methods. For the given optimization space, if there is restriction towards any method, add the method in this input with the name as the key and bool False as the value. Should leave empty if no speficied restriction needed.
            **params (optional): Should contain all other information for the specified econometric analysis method.
        """
        
        # Initialize the attributes
        self.dependent_variable = dependent_variable
        self.treatment_variable = treatment_variable
        self.treatment_is_dummy = treatment_is_dummy
        self.covariance_requirement = covariance_requirement
        self.control_variables = control_variables
        self.categorial_control_name = categorial_control_name
        self.trimming_requirement = trimming_requirement
        self.params_dict = params
        
        # Categorical variables cannot be directly figured out, but at least dummy variables can
        if self.control_variables is not None:
            for each_control in self.control_variables.columns:
                if self.control_variables[each_control].unique().shape[0] == 2 and each_control not in self.categorial_control_name:
                    print("Control Variable:", each_control, "is dummy but not included in categorical control name list. Has been added now!")
                    self.categorial_control_name.append(each_control)
    
        # Set the optimization plan
        optimization_plan = {
            "dependent_variable_transformation": True, 
            "treatment_variable_transformation": True, 
            "control_variable_selection": True, 
            "control_variable_transformation": True, 
            "sample_trimming_test": True, 
            "covariance_requirement": covariance_requirement
            }
        
        # Check if any optimization method cannot be implemented
        if self.treatment_is_dummy is True:
            optimization_plan["treatment_variable_transformation"] = False
        if self.control_variables is None:
            optimization_plan["control_variable_selection"] = False
            optimization_plan["control_variable_transformation"] = False
        if self.trimming_requirement is not None:
            optimization_plan["sample_trimming_test"] = False
        
        # If user specifies some particular methods not to implement, adjust the plan, and then finalize the plan
        for each_restriction in restriction_dict:
            if each_restriction in optimization_plan:
                optimization_plan[each_restriction] = False
            else:
                raise RuntimeError("Specified restriction does not direct to an optimization method. Please check!")
        self.plan = optimization_plan
        
        # Store the best setting after optimization
        self.best_setting = None
        
    # -------------------------------------------------------------------------
    
    def optimization(self, target_type: str = "neg_pvalue"):
        
        """
        Generate the optimization parametric space and optimize the final analysis result.
        Also provide optimization parametric sets for top 10 results.
        
        Args:
            target_type (str, optional): An instruction about the way to evaluate the regression outcome. When set default as "neg_pvalue", the evaluation criteria is the negative value of treatment variable's coefficient p-value. Can also choose "rsquared" which output the adjusted R-squared value for the second-step regression.
        """
        
        # Check the evaluation target input
        if target_type not in ["neg_pvalue", "rsquared"]:
            raise RuntimeError("Evaluation Target Type Not Supported, Please Check!")
        
        # Generate detailed parametric sets based on the plan
        detailed_parametric_space = {}
        
        # Check towards dependent variable transformation
        if self.plan["dependent_variable_transformation"] is True:
            if self.dependent_variable[self.dependent_variable <= 0].shape[0] == 0:
                detailed_parametric_space["dependent_variable_transformation"] = ["original", "log", "remove_extreme"]
            else:
                detailed_parametric_space["dependent_variable_transformation"] = ["original", "remove_extreme"]
        else:
            detailed_parametric_space["dependent_variable_transformation"] = ["original"]
        
        # Check towards treatment variable transformation
        if self.plan["treatment_variable_transformation"] is True:
            if self.treatment_variable[self.treatment_variable <= 0].shape[0] == 0:
                detailed_parametric_space["treatment_variable_transformation"] = ["original", "log", "remove_extreme", "standardization", "remove_extreme_standardization"]
            else:
                detailed_parametric_space["treatment_variable_transformation"] = ["original", "remove_extreme", "standardization", "remove_extreme_standardization"]
        else:
            detailed_parametric_space["treatment_variable_transformation"] = ["original"]

        # Check towards control variable selection
        if self.plan["control_variable_selection"] is True:
            all_control_variable_amount = self.control_variables.shape[1]
            all_control_variable_list = list(self.control_variables.columns)
            all_possible_control_variable_combinations = []
            # all_possible_control_variable_combinations.append([])  # In the propensity score method, control variable must not be None value! Therefore, if control variable selection is allowed by user, there should not be the case that all control variables are abandoned.
            for i in range(1, all_control_variable_amount + 1):
                for ii in combinations(all_control_variable_list, i):
                    all_possible_control_variable_combinations.append(list(ii))
            detailed_parametric_space["control_variable_selection"] = all_possible_control_variable_combinations            
        else:
            if self.control_variables is not None:
                detailed_parametric_space["control_variable_selection"] = [list(self.control_variables.columns)]
            else:
                detailed_parametric_space["control_variable_selection"] = [[]]
            
        # Check towards control variable transformation
        if self.plan["control_variable_transformation"] is True:
            control_variable_transformation_dict = {}
            for each_control_variable in self.control_variables.columns:
                if each_control_variable in self.categorial_control_name:
                    control_variable_transformation_dict[each_control_variable] = ["original"]
                else:
                    if self.control_variables[each_control_variable][self.control_variables[each_control_variable] <= 0].shape[0] == 0:
                        control_variable_transformation_dict[each_control_variable] = ["original", "log", "standardization", "mid_value_dummy"]
                    else:
                        control_variable_transformation_dict[each_control_variable] = ["original", "standardization", "mid_value_dummy"]
            detailed_parametric_space["control_variable_transformation"] = control_variable_transformation_dict
        else:
            control_variable_transformation_dict = {}
            if self.control_variables is not None:
                for each_control_variable in self.control_variables.columns:
                    control_variable_transformation_dict[each_control_variable] = ["original"]
            detailed_parametric_space["control_variable_transformation"] = control_variable_transformation_dict

        # Check towards sample trimming test
        if self.plan["sample_trimming_test"] is True:
            detailed_parametric_space["sample_trimming_test"] = [None, [0.05, 0.95], [0.1, 0.9], [0.15, 0.85], [0.2, 0.8]]
        else:
            detailed_parametric_space["sample_trimming_test"] = [self.trimming_requirement]

        # Check towards cov requirement
        if self.plan["covariance_requirement"] is True:
            detailed_parametric_space["covariance_requirement"] = ["HC0", "HC1", "HC2", "HC3"]    
        else:
            detailed_parametric_space["covariance_requirement"] = [None]
            
        # ---------------------------------------------------------------------
        
        # Adjust the transformation data type (from a dict of list of str to a list of dict)
        detailed_parametric_space["control_variable_transformation"] = self.dictlist_listdict_transformation(detailed_parametric_space["control_variable_transformation"])
        
        # Record all results
        all_setting_dict_record_list = []
        all_setting_dict_record = {}
        count = 0
        for each_dependent_variable_transformation in detailed_parametric_space["dependent_variable_transformation"]:
            for each_treatment_variable_transformation in detailed_parametric_space["treatment_variable_transformation"]:
                for each_control_variable_selection in detailed_parametric_space["control_variable_selection"]:
                    for each_control_variable_transformation in detailed_parametric_space["control_variable_transformation"]:
                        for each_sample_trimming_test in detailed_parametric_space["sample_trimming_test"]:
                            for each_covariance_requirement in detailed_parametric_space["covariance_requirement"]:
                                each_control_variable_transformation = {i: each_control_variable_transformation[i] for i in each_control_variable_selection}
                                temp_dict = {
                                    "dependent_variable_transformation": each_dependent_variable_transformation, 
                                    "treatment_variable_transformation": each_treatment_variable_transformation, 
                                    "control_variable_selection": each_control_variable_selection, 
                                    "control_variable_transformation": each_control_variable_transformation, 
                                    "sample_trimming_test": each_sample_trimming_test, 
                                    "covariance_requirement": each_covariance_requirement
                                    }
                                if temp_dict not in all_setting_dict_record_list:
                                    count += 1
                                    all_setting_dict_record[count] = temp_dict
                                    all_setting_dict_record_list.append(temp_dict)
        all_results_series = pd.Series(index = all_setting_dict_record.keys())
        print("Total Optimization Trials:", all_results_series.shape[0])
            
        # ---------------------------------------------------------------------
        
        # Go through each setting, run the regression and store the result
        print("Starting to Conduct Optimization! Will need some time...")
        for each_setting_count in all_setting_dict_record:
            
            # Process each setting treatment
            dependent_variable = self.process_a_data_series(self.dependent_variable, all_setting_dict_record[each_setting_count]["dependent_variable_transformation"])
            treatment_variable = self.process_a_data_series(self.treatment_variable, all_setting_dict_record[each_setting_count]["treatment_variable_transformation"])
            if self.control_variables is not None:
                control_variables = self.control_variables[all_setting_dict_record[each_setting_count]["control_variable_selection"]]
                if control_variables.shape[1] == 0:
                    control_variables = None
                else:
                    for each_control in control_variables:
                        control_variables.loc[:, each_control] = self.process_a_data_series(control_variables[each_control], all_setting_dict_record[each_setting_count]["control_variable_transformation"][each_control]).copy()
            else:
                control_variables = None
            sample_trimming = all_setting_dict_record[each_setting_count]["sample_trimming_test"]
            cov_type = all_setting_dict_record[each_setting_count]["covariance_requirement"]
            
            # Since dependent variable and treatment variable might have remove extreme operation, should remember to match data after all transformations
            dependent_variable = dependent_variable[dependent_variable.index.isin(treatment_variable.index)]
            treatment_variable = treatment_variable.loc[dependent_variable.index]
            if control_variables is not None:
                control_variables = control_variables.loc[dependent_variable.index]
            
            # Run the regression, obtain and store the result
            propensity_score = propensity_score_construction(treatment_variable, control_variables)
            final_result = propensity_score_regression(dependent_variable, treatment_variable, propensity_score, sample_trimming = sample_trimming, cov_type = cov_type, target_type = target_type, output_tables = False)
            all_results_series[each_setting_count] = final_result
    
        # Print out the top 10 result settings
        all_results_series = all_results_series.sort_values(ascending = False)
        if all_results_series.shape[0] < 10:
            selected_range = list(range(1, all_results_series.shape[0] + 1))
        else:
            selected_range = list(range(1, 10 + 1))
        print("==============================================================")
        print("The Best 10 Results and Settings:")
        print("--------------------------------------------------------------")
        for i in selected_range:
            selected_index = all_results_series.index[i - 1]
            print("--------------------------------------------------------------")
            print("Count:", i)
            print("Result:", all_results_series.iloc[i - 1])
            print("Setting Information:", all_setting_dict_record[selected_index])
            print("--------------------------------------------------------------")
        print("==============================================================")

        # Output the best result
        selected_setting_count = all_results_series.index[0]
        
        # Process each setting treatment
        dependent_variable = self.process_a_data_series(self.dependent_variable, all_setting_dict_record[selected_setting_count]["dependent_variable_transformation"])
        treatment_variable = self.process_a_data_series(self.treatment_variable, all_setting_dict_record[selected_setting_count]["treatment_variable_transformation"])
        if self.control_variables is not None:
            control_variables = self.control_variables[all_setting_dict_record[selected_setting_count]["control_variable_selection"]]
            if control_variables.shape[1] == 0:
                control_variables = None
            else:
                for each_control in control_variables:
                    control_variables.loc[:, each_control] = self.process_a_data_series(control_variables[each_control], all_setting_dict_record[selected_setting_count]["control_variable_transformation"][each_control]).copy()
        else:
            control_variables = None
        sample_trimming = all_setting_dict_record[selected_setting_count]["sample_trimming_test"]
        cov_type = all_setting_dict_record[selected_setting_count]["covariance_requirement"]
            
        # Since dependent variable and treatment variable might have remove extreme operation, should remember to match data after all transformations
        dependent_variable = dependent_variable[dependent_variable.index.isin(treatment_variable.index)]
        treatment_variable = treatment_variable.loc[dependent_variable.index]
        if control_variables is not None:
            control_variables = control_variables.loc[dependent_variable.index]
        
        # Run the regression
        propensity_score = propensity_score_construction(treatment_variable, control_variables)
        final_result = propensity_score_regression(dependent_variable, treatment_variable, propensity_score, sample_trimming = sample_trimming, cov_type = cov_type, target_type = None, output_tables = True)
        
        # Store the best setting
        self.best_setting = all_setting_dict_record[selected_setting_count]
    
#%%

@register_tool(tags=["econometric optimization"])
class IV2SLSOptimization(GeneralOptimizationMethods):
    """
    Optimize the IV-2SLS regression result in the given optimization parametric space.
    """
    
    def __init__(self, 
                 dependent_variable: pd.Series, 
                 treatment_variable: pd.Series, 
                 treatment_is_dummy: bool, 
                 IV_variable, 
                 control_variables = None, 
                 covariance_requirement: bool = False, 
                 categorial_IV_name: list = [], 
                 categorial_control_name: list = [], 
                 restriction_dict: dict = {}, 
                 **params):
        
        """
        Initialize self and generate the optimization plan. The input data Series or DataFrames should not contain any nan value.
        The optimization space includes "dependent_variable_transformation", "treatment_variable_transformation", "control_variable_selection", 
        "control_variable_transformation", "IV_selection", "IV_transformation" and finally "covariance_requirement". WHEN CALLING THIS OBJECT, 
        BE SURE USER's RESTRICTIONS (IF ANY) ARE CAREFULLY CONTROLLED IN THE ARGUMENT 'restriction_dict'. If any aspect is not allowed by the user,
        put the aspect name into the argument 'restriction_dict' as the key and bool False as the value.
        After successfully calling the attribute function "optimization", you can obtain the best optimization setting via the attribute variable "self.best_setting".

        Args:
            dependent_variable (pd.Series): Target dependent variable, which now is required to be a continuous numerical variable.
            treatment_variable (pd.Series): Target treatment variable, which should be a dummy or continuous variable.
            treatment_is_dummy (bool): Denote if treatment variable is not continuous but a CATEGORICAL OR DUMMY variable. 
            IV_variable (pd.Series or pd.DataFrame): Proposed instrument variable(s). Could have only one or multiple IVs.
            control_variables (pd.DataFrame or None, optional): A dataframe of control variables, which should not contain the intercept. If there is no specified control variable, leave it the default None value. 
            covariance_requirement (bool, optional): Specify if the IV-2SLS regression method requires COVARIANCE adjustment. If required, this aspect will also be included for optimization. Default is False.
            categorial_IV_name (list, optional): A list of IV names, each of which in this list indicate the denoted IV IS NOT CONTINUOUS BUT A CATEGORICAL OR DUMMY variable.
            categorial_control_name (list, optional): A list of control variable names, each of which in this list indicate the denoted control variable is not continuous but a categorial variable.
            restriction_dict (dict, optional): Detailed restrictions towards the optimization methods. For the given optimization space, if there is restriction towards any method, add the method in this input with the name as the key and bool False as the value. Should leave empty if no speficied restriction needed.
            **params (optional): Should contain all other information for the specified econometric analysis method.
        """
        
        # Initialize the attributes
        self.dependent_variable = dependent_variable
        self.treatment_variable = treatment_variable
        self.treatment_is_dummy = treatment_is_dummy
        self.IV_variable = pd.DataFrame(IV_variable)
        self.covariance_requirement = covariance_requirement
        self.control_variables = control_variables
        self.categorial_IV_name = categorial_IV_name
        self.categorial_control_name = categorial_control_name
        self.params_dict = params
        
        # Categorical variables cannot be directly figured out, but at least dummy variables can
        for each_IV in self.IV_variable.columns:
            if self.IV_variable[each_IV].unique().shape[0] == 2 and each_IV not in self.categorial_IV_name:
                print("IV Variable:", each_IV, "is dummy but not included in categorical IV name list. Has been added now!")
                self.categorial_IV_name.append(each_IV)
        if self.control_variables is not None:
            for each_control in self.control_variables.columns:
                if self.control_variables[each_control].unique().shape[0] == 2 and each_control not in self.categorial_control_name:
                    print("Control Variable:", each_control, "is dummy but not included in categorical control name list. Has been added now!")
                    self.categorial_control_name.append(each_control)
    
        # Set the optimization plan
        optimization_plan = {
            "dependent_variable_transformation": True, 
            "treatment_variable_transformation": True, 
            "control_variable_selection": True, 
            "control_variable_transformation": True, 
            "IV_selection": True, 
            "IV_transformation": True, 
            "covariance_requirement": covariance_requirement
            }
        
        # Check if any optimization method cannot be implemented
        if self.treatment_is_dummy is True:
            optimization_plan["treatment_variable_transformation"] = False
        if self.control_variables is None:
            optimization_plan["control_variable_selection"] = False
            optimization_plan["control_variable_transformation"] = False
        if self.IV_variable.shape[1] == 1:
            optimization_plan["IV_selection"] = False
        
        # If user specifies some particular methods not to implement, adjust the plan, and then finalize the plan
        for each_restriction in restriction_dict:
            if each_restriction in optimization_plan:
                optimization_plan[each_restriction] = False
            else:
                raise RuntimeError("Specified restriction does not direct to an optimization method. Please check!")
        self.plan = optimization_plan
        
        # Store the best setting after optimization
        self.best_setting = None
        
    # -------------------------------------------------------------------------
        
    def optimization(self, target_type: str = "neg_pvalue", check_IV_validity: bool = False, check_process_weak_IV: bool = False):
        
        """
        Generate the optimization parametric space and optimize the final analysis result.
        Also provide optimization parametric sets for top 10 results.
        
        Args:
            target_type (str, optional): An instruction about the way to evaluate the regression outcome. When set default as "neg_pvalue", the evaluation criteria is the negative value of treatment variable's coefficient p-value. Can also choose "rsquared" which output the adjusted R-squared value for the second-step regression.
            check_IV_validity (bool, optional): Denote if the validity of each IV should be considered into the final evaluation. Set default to be False since this function is not yet implemented.
            check_process_weak_IV (bool, optional): Denote if the optimization space include the processing method of weak IVs. Set default to be False since this function is not yet implemented.
        """
        
        # Check the evaluation target input
        if target_type not in ["neg_pvalue", "rsquared"]:
            raise RuntimeError("Evaluation Target Type Not Supported, Please Check!")
        
        # Generate detailed parametric sets based on the plan
        detailed_parametric_space = {}
        
        # Check towards dependent variable transformation
        if self.plan["dependent_variable_transformation"] is True:
            if self.dependent_variable[self.dependent_variable <= 0].shape[0] == 0:
                detailed_parametric_space["dependent_variable_transformation"] = ["original", "log", "remove_extreme"]
            else:
                detailed_parametric_space["dependent_variable_transformation"] = ["original", "remove_extreme"]
        else:
            detailed_parametric_space["dependent_variable_transformation"] = ["original"]
        
        # Check towards treatment variable transformation
        if self.plan["treatment_variable_transformation"] is True:
            if self.treatment_variable[self.treatment_variable <= 0].shape[0] == 0:
                detailed_parametric_space["treatment_variable_transformation"] = ["original", "log", "remove_extreme", "standardization", "remove_extreme_standardization"]
            else:
                detailed_parametric_space["treatment_variable_transformation"] = ["original", "remove_extreme", "standardization", "remove_extreme_standardization"]
        else:
            detailed_parametric_space["treatment_variable_transformation"] = ["original"]

        # Check towards IV selection
        if self.plan["IV_selection"] is True:
            all_IV_amount = self.IV_variable.shape[1]
            all_IV_list = list(self.IV_variable.columns)
            all_possible_IV_combinations = []
            for i in range(1, all_IV_amount + 1):
                for ii in combinations(all_IV_list, i):
                    all_possible_IV_combinations.append(list(ii))
            detailed_parametric_space["IV_selection"] = all_possible_IV_combinations
        else:
            detailed_parametric_space["IV_selection"] = [list(self.IV_variable.columns)]
            
        # Check towards IV transformation
        if self.plan["IV_transformation"] is True:
            IV_transformation_dict = {}
            for each_IV_variable in self.IV_variable.columns:
                if each_IV_variable in self.categorial_IV_name:
                    IV_transformation_dict[each_IV_variable] = ["original"]
                else:
                    if self.IV_variable[each_IV_variable][self.IV_variable[each_IV_variable] <= 0].shape[0] == 0:
                        IV_transformation_dict[each_IV_variable] = ["original", "log", "standardization", "mid_value_dummy"]
                    else:
                        IV_transformation_dict[each_IV_variable] = ["original", "standardization", "mid_value_dummy"]
            detailed_parametric_space["IV_transformation"] = IV_transformation_dict
        else:
            IV_transformation_dict = {}
            for each_IV_variable in self.IV_variable.columns:
                IV_transformation_dict[each_IV_variable] = ["original"]
            detailed_parametric_space["IV_transformation"] = IV_transformation_dict

        # Check towards control variable selection
        if self.plan["control_variable_selection"] is True:
            all_control_variable_amount = self.control_variables.shape[1]
            all_control_variable_list = list(self.control_variables.columns)
            all_possible_control_variable_combinations = []
            all_possible_control_variable_combinations.append([])
            for i in range(1, all_control_variable_amount + 1):
                for ii in combinations(all_control_variable_list, i):
                    all_possible_control_variable_combinations.append(list(ii))
            detailed_parametric_space["control_variable_selection"] = all_possible_control_variable_combinations            
        else:
            if self.selected_setting_count is not None:
                detailed_parametric_space["control_variable_selection"] = [list(self.control_variables.columns)]
            else:
                detailed_parametric_space["control_variable_selection"] =[[]]
            
        # Check towards control variable transformation
        if self.plan["control_variable_transformation"] is True:
            control_variable_transformation_dict = {}
            for each_control_variable in self.control_variables.columns:
                if each_control_variable in self.categorial_control_name:
                    control_variable_transformation_dict[each_control_variable] = ["original"]
                else:
                    if self.control_variables[each_control_variable][self.control_variables[each_control_variable] <= 0].shape[0] == 0:
                        control_variable_transformation_dict[each_control_variable] = ["original", "log", "standardization", "mid_value_dummy"]
                    else:
                        control_variable_transformation_dict[each_control_variable] = ["original", "standardization", "mid_value_dummy"]
            detailed_parametric_space["control_variable_transformation"] = control_variable_transformation_dict
        else:
            control_variable_transformation_dict = {}
            if self.control_variables is not None:
                for each_control_variable in self.control_variables.columns:
                    control_variable_transformation_dict[each_control_variable] = ["original"]
            detailed_parametric_space["control_variable_transformation"] = control_variable_transformation_dict

        # Check towards cov requirement
        if self.plan["covariance_requirement"] is True:
            detailed_parametric_space["covariance_requirement"] = ["HC0", "HC1", "HC2", "HC3"]    
        else:
            detailed_parametric_space["covariance_requirement"] = [None]
    
        # ---------------------------------------------------------------------
        
        # Adjust the transformation data type (from a dict of list of str to a list of dict)
        detailed_parametric_space["IV_transformation"] = self.dictlist_listdict_transformation(detailed_parametric_space["IV_transformation"])
        detailed_parametric_space["control_variable_transformation"] = self.dictlist_listdict_transformation(detailed_parametric_space["control_variable_transformation"])
        
        # Record all results
        all_setting_dict_record_list = []
        all_setting_dict_record = {}
        count = 0
        for each_dependent_variable_transformation in detailed_parametric_space["dependent_variable_transformation"]:
            for each_treatment_variable_transformation in detailed_parametric_space["treatment_variable_transformation"]:
                for each_IV_selection in detailed_parametric_space["IV_selection"]:
                    for each_IV_transformation in detailed_parametric_space["IV_transformation"]:
                        for each_control_variable_selection in detailed_parametric_space["control_variable_selection"]:
                            for each_control_variable_transformation in detailed_parametric_space["control_variable_transformation"]:
                                for each_covariance_requirement in detailed_parametric_space["covariance_requirement"]:
                                    each_IV_transformation = {i: each_IV_transformation[i] for i in each_IV_selection}
                                    each_control_variable_transformation = {i: each_control_variable_transformation[i] for i in each_control_variable_selection}
                                    temp_dict = {
                                        "dependent_variable_transformation": each_dependent_variable_transformation, 
                                        "treatment_variable_transformation": each_treatment_variable_transformation, 
                                        "IV_selection": each_IV_selection, 
                                        "IV_transformation": each_IV_transformation, 
                                        "control_variable_selection": each_control_variable_selection, 
                                        "control_variable_transformation": each_control_variable_transformation, 
                                        "covariance_requirement": each_covariance_requirement
                                        }
                                    if temp_dict not in all_setting_dict_record_list:
                                        count += 1
                                        all_setting_dict_record[count] = temp_dict
                                        all_setting_dict_record_list.append(temp_dict)
        all_results_series = pd.Series(index = all_setting_dict_record.keys())
        print("Total Optimization Trials:", all_results_series.shape[0])
            
        # ---------------------------------------------------------------------
        
        # Go through each setting, run the regression and store the result
        print("Starting to Conduct Optimization! Will need some time...")
        for each_setting_count in all_setting_dict_record:
            
            # Process each setting treatment
            dependent_variable = self.process_a_data_series(self.dependent_variable, all_setting_dict_record[each_setting_count]["dependent_variable_transformation"])
            treatment_variable = self.process_a_data_series(self.treatment_variable, all_setting_dict_record[each_setting_count]["treatment_variable_transformation"])
            IV_variable = self.IV_variable[all_setting_dict_record[each_setting_count]["IV_selection"]]
            for each_IV in IV_variable:
                IV_variable.loc[:, each_IV] = self.process_a_data_series(IV_variable[each_IV], all_setting_dict_record[each_setting_count]["IV_transformation"][each_IV]).copy()
            if self.control_variables is not None:
                control_variables = self.control_variables[all_setting_dict_record[each_setting_count]["control_variable_selection"]]
                if control_variables.shape[1] == 0:
                    control_variables = None
                else:
                    for each_control in control_variables:
                        control_variables.loc[:, each_control] = self.process_a_data_series(control_variables[each_control], all_setting_dict_record[each_setting_count]["control_variable_transformation"][each_control]).copy()
            else:
                control_variables = None
            cov_type = all_setting_dict_record[each_setting_count]["covariance_requirement"]
            
            # Since dependent variable and treatment variable might have remove extreme operation, should remember to match data after all transformations
            dependent_variable = dependent_variable[dependent_variable.index.isin(treatment_variable.index)]
            treatment_variable = treatment_variable.loc[dependent_variable.index]
            IV_variable = IV_variable.loc[dependent_variable.index]
            if control_variables is not None:
                control_variables = control_variables.loc[dependent_variable.index]
            
            # Run the regression, obtain and store the result
            final_result = IV_2SLS_regression(dependent_variable, treatment_variable, IV_variable, control_variables, cov_type, target_type = target_type, output_tables = False)
            all_results_series[each_setting_count] = final_result
    
        # Print out the top 10 result settings
        all_results_series = all_results_series.sort_values(ascending = False)
        if all_results_series.shape[0] < 10:
            selected_range = list(range(1, all_results_series.shape[0] + 1))
        else:
            selected_range = list(range(1, 10 + 1))
        print("==============================================================")
        print("The Best 10 Results and Settings:")
        print("--------------------------------------------------------------")
        for i in selected_range:
            selected_index = all_results_series.index[i - 1]
            print("--------------------------------------------------------------")
            print("Count:", i)
            print("Result:", all_results_series.iloc[i - 1])
            print("Setting Information:", all_setting_dict_record[selected_index])
            print("--------------------------------------------------------------")
        print("==============================================================")

        # Output the best result
        selected_setting_count = all_results_series.index[0]
        
        # Process each setting treatment
        dependent_variable = self.process_a_data_series(self.dependent_variable, all_setting_dict_record[selected_setting_count]["dependent_variable_transformation"])
        treatment_variable = self.process_a_data_series(self.treatment_variable, all_setting_dict_record[selected_setting_count]["treatment_variable_transformation"])
        IV_variable = self.IV_variable[all_setting_dict_record[selected_setting_count]["IV_selection"]]
        for each_IV in IV_variable:
            IV_variable.loc[:, each_IV] = self.process_a_data_series(IV_variable[each_IV], all_setting_dict_record[selected_setting_count]["IV_transformation"][each_IV]).copy()
        if self.control_variables is not None:
            control_variables = self.control_variables[all_setting_dict_record[selected_setting_count]["control_variable_selection"]]
            if control_variables.shape[0] == 0:
                control_variables = None
            else:
                for each_control in control_variables:
                    control_variables.loc[:, each_control] = self.process_a_data_series(control_variables[each_control], all_setting_dict_record[selected_setting_count]["control_variable_transformation"][each_control]).copy()
        else:
            control_variables = None
        cov_type = all_setting_dict_record[selected_setting_count]["covariance_requirement"]
            
        # Since dependent variable and treatment variable might have remove extreme operation, should remember to match data after all transformations
        dependent_variable = dependent_variable[dependent_variable.index.isin(treatment_variable.index)]
        treatment_variable = treatment_variable.loc[dependent_variable.index]
        IV_variable = IV_variable.loc[dependent_variable.index]
        if control_variables is not None:
            control_variables = control_variables.loc[dependent_variable.index]
            
        # Run the regression
        IV_2SLS_regression(dependent_variable, treatment_variable, IV_variable, control_variables, cov_type, target_type = None, output_tables = True)
        
        # Store the best setting
        self.best_setting = all_setting_dict_record[selected_setting_count]

#%%

@register_tool(tags=["econometric optimization"])
class DID_Optimization(GeneralOptimizationMethods):
    pass  # TODO

#%%

@register_tool(tags=["econometric optimization"])
class RDD_Optimization(GeneralOptimizationMethods):
    pass  # TODO

#%%
#%%
#%%
#%%
#%%
#%%

@register_tool(tags=["econometric optimization"])
class SC_DID_Optimization(GeneralOptimizationMethods):
    pass  # TODO

#%%

@register_tool(tags=["econometric optimization"])
class SC_RDD_Optimization(GeneralOptimizationMethods):
    pass  # TODO

#%%

@register_tool(tags=["econometric optimization"])
class PSM_DID_Optimization(GeneralOptimizationMethods):
    pass  # TODO

#%%

@register_tool(tags=["econometric optimization"])
class PSM_RDD_Optimization(GeneralOptimizationMethods):
    pass  # TODO