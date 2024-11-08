from __future__ import annotations
import sys
sys.path.append("/home/tianyang/ChatInterpreter/ML_Assistant")
from metagpt.tools.tool_registry import register_tool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels import PanelOLS

#%%

@register_tool(tags=["econometric algorithm"])
def ordinary_least_square_regression(dependent_variable, treatment_variable, covariate_variables, cov_type = None, target_type = None, output_tables = False):
    
    """
    Use Ordinary Least Square Regression method to estimate Average Treatment Effect (ATE) of 
    the treatment variable towards the dependent variable.
    NOTE THAT THIS FUNCTION DOES NOT RETURN THE FINAL REGRESSION TABLE! All tables can (and only can) be printed out during the function.
    The final return is some clearly specified parameter or statistic within the regressions.
    
    Args:
        dependent_variable (pd.Series): Target dependent variable, which should not contain nan value.
        treatment_variable (pd.Series): Target treatment variable, which should not contain nan value.
        covariate_variables (pd.DataFrame or None): Proposed covariate variables. If user does not specify any covariate variable, this could be None. Otherwise, it should not contain nan value.
        cov_type (str or None): The covariance estimator used in the results. If not specified by user, this could be None.
        target_type (str or None): Denote whether this function need to return any specific evaluation metric or any other content. If only want to output regression tables, this should be None. Also three possible inputs, "neg_pvalue" for the regression treatment variable coefficient p-value's negative value, "rsquared" for the adjusted R-squared value of the regression, and "final_model" for the final regression model.
        output_tables (bool): Denote whether this function need to print out regression tables. If want to see the tabels, this should be True. If only want the evaluation metric outputs, this should be False.
    """
    
    # Run the regression
    if covariate_variables is None:
        X = treatment_variable
    else:
        X = pd.concat([treatment_variable, covariate_variables], axis = 1)
    if cov_type is None:
        regression = sm.OLS(dependent_variable, sm.add_constant(X)).fit()
    else:
        regression = sm.OLS(dependent_variable, sm.add_constant(X)).fit(cov_type = cov_type)
    
    # Output the final result. ATE is the coefficient of the predicted treatment variable
    if output_tables is True:
        print(regression.summary())

    # Return evaluation metric if needed
    if target_type == "neg_pvalue":
        return -regression.pvalues[treatment_variable.name]
    elif target_type == "rsquared":
        return regression.rsquared_adj
    elif target_type == "final_model":
        return regression
    
@register_tool(tags=["econometric algorithm"])
def Panel_Data_OLS_regression(dependent_variable, treatment_variable, covariate_variables, entity_effect = False, time_effect = False, other_effect = None, cov_type = "unadjusted", target_type = None, output_tables = False):
    
    """
    Use Ordinary Least Square Regression method to estimate Average Treatment Effect (ATE) of 
    the treatment variable towards the dependent variable, in the PANEL DATA format. This function also integrates PooledOLS method altogether.
    NOTE THAT THIS FUNCTION DOES NOT RETURN THE FINAL REGRESSION TABLE! All tables can (and only can) be printed out during the function.
    The final return is some clearly specified parameter or statistic within the regressions.
    
    Args:
        dependent_variable (pd.Series): Target dependent variable, which should not contain nan value. The index of the series should be entity-time multi-index.
        treatment_variable (pd.Series): Target treatment variable, which should not contain nan value. The index of the series should be entity-time multi-index.
        covariate_variables (pd.DataFrame or None): Proposed covariate variables. If user does not specify any covariate variable, this could be None. Otherwise, it should not contain nan value.
        entity_effect (bool): Denote whether entity effect is included in the PanelOLS regression.
        time_effect (bool): Denote whether time effect is included in the PanelOLS regression.
        other_effect (pd.DataFrame or None): Denote whether other effects are included in the PanelOLS regression. If there are other effects required, this input should be a pd.DataFrame with the categorial variable column(s) and entity-time multi-index. If no other effects required, leave this input to be None.
        cov_type (str): The covariance estimator used in the results. Five covariance estimators are supported: "unadjusted" for homoskedastic residual, "robust" for heteroskedasticity control, "cluster_entity" for entity clustering, "cluster_time" for time clustering, and "cluster_both" for entity-time two-way clustering.
        target_type (str or None): Denote whether this function need to return any specific evaluation metric or any other content. If only want to output regression tables, this should be None. Also three possible inputs, "neg_pvalue" for the regression treatment variable coefficient p-value's negative value, "rsquared" for the adjusted R-squared value of the regression, and "final_model" for the final regression model.
        output_tables (bool): Denote whether this function need to print out regression tables. If want to see the tabels, this should be True. If only want the evaluation metric outputs, this should be False.
    """

    # Check if inputs are proper formatted
    if cov_type not in ["unadjusted", "robust", "cluster_entity", "cluster_time", "cluster_both"]:
        raise RuntimeError("Covariance type input unsupported! This function supports 'unadjusted', 'robust', 'cluster_entity', 'cluster_time' and 'cluster_both' as possible inputs!")
    count_effects = 0
    if entity_effect is True:
        count_effects += 1
    if time_effect is True:
        count_effects += 1
    if other_effect is not None:
        count_effects += other_effect.shape[1]
    if count_effects > 2:
        raise RuntimeError("At most two effects allowed! Please note that now there are " + str(count_effects) + " effects in total!")

    # Prepare the dataset
    if covariate_variables is None:
        X = treatment_variable
    else:
        X = pd.concat([treatment_variable, covariate_variables], axis = 1)
    if count_effects == 0:
        X = sm.add_constant(X)
    
    # Run the regression
    if cov_type in ["unadjusted", "robust"]:
        regression = PanelOLS(dependent_variable, X, entity_effect = entity_effect, time_effect = time_effect, other_effect = other_effect).fit(cov_type = cov_type)
    elif cov_type == "cluster_entity":
        regression = PanelOLS(dependent_variable, X, entity_effect = entity_effect, time_effect = time_effect, other_effect = other_effect).fit(cov_type = "clustered", cluster_entity = True)
    elif cov_type == "cluster_time":
        regression = PanelOLS(dependent_variable, X, entity_effect = entity_effect, time_effect = time_effect, other_effect = other_effect).fit(cov_type = "clustered", cluster_time = True)
    elif cov_type == "cluster_both":
        regression = PanelOLS(dependent_variable, X, entity_effect = entity_effect, time_effect = time_effect, other_effect = other_effect).fit(cov_type = "clustered", cluster_entity = True, cluster_time = True)

    # Output the final result. ATE is the coefficient of the predicted treatment variable
    if output_tables is True:
        print(regression)

    # Return evaluation metric if needed
    if target_type == "neg_pvalue":
        return -regression.pvalues[treatment_variable.name]
    elif target_type == "rsquared":
        return regression.rsquared
    elif target_type == "final_model":
        return regression
    
#%%

@register_tool(tags=["econometric algorithm"])
def propensity_score_construction(treatment_variable, covariate_variable):
    
    """
    Construct propensity score for each sample to receive binary treatment based on covariate variables, using binary Logistic regression.
    
    Args:
        treatment_variable (pd.Series): Target treatment variable, which should be a binary variable (1 for treatment, 0 for control).
        covariate_variable (pd.DataFrame): A dataframe of covariate variables, which should not contain nan value or intercept.

    Returns:
        pd.Series: The estimated propensity score for each sample, which will be named "propensity_score".
    """
    
    # Directly apply Logistic regression method to estimate the propensity score
    clf = sm.Logit(treatment_variable, sm.add_constant(covariate_variable)).fit()
    result_series = pd.Series(clf.predict(sm.add_constant(covariate_variable)), index = covariate_variable.index)
    result_series.name = "propensity_score"
    return result_series

@register_tool(tags=["econometric algorithm"])
def propensity_score_regression(dependent_variable, treatment_variable, propensity_score, sample_trimming = None, cov_type = None, target_type = None, output_tables = False):
    
    """
    Use propensity score method to estimate Average Treatment Effect (ATE) of the treatment variable towards the dependent variable. 
    If user specifies to trim samples, this tool will implement sample trimming based on detailed parameters.
    The ATE is the coefficient of the target treatment variable in the final OLS regression.
    NOTE THAT THIS FUNCTION DOES NOT RETURN THE FINAL REGRESSION TABLE! All tables can (and only can) be printed out during the function.
    The final return is some clearly specified parameter or statistic within the regressions.
    
    Args:
        dependent_variable (pd.Series): Target dependent variable, which should not contain nan value.
        treatment_variable (pd.Series): Target treatment variable, which should be a binary variable with no nan value (1 for treatment, 0 for control).
        propensity_score (pd.Series): Propensity score for each sample to receive treatment, which should not contain nan value.
        sample_trimming (list or None): A list containing propensity-score-based sample trimming requirement, for example, [0.05, 0.95]. This input should be left None if no sample trimming is required.
        cov_type (str or None): The covariance estimator used in the results. If not specified by user, this could be None.
        target_type (str or None): Denote whether this function need to return any specific evaluation metric or any other content. If only want to output regression tables, this should be None. Also three possible inputs, "neg_pvalue" for the regression treatment variable coefficient p-value's negative value, "rsquared" for the adjusted R-squared value of the regression, and "final_model" for the final regression model.
        output_tables (bool): Denote whether this function need to print out regression tables. If want to see the tabels, this should be True. If only want the evaluation metric outputs, this should be False.
    """
    
    # Conduct sample trimming when required
    if sample_trimming is not None:
        propensity_score = propensity_score[propensity_score >= sample_trimming[0]]
        propensity_score = propensity_score[propensity_score <= sample_trimming[1]]
        dependent_variable = dependent_variable.loc[propensity_score.index]
        treatment_variable = treatment_variable.loc[propensity_score.index]
        
    # Run the OLS regression
    if cov_type is None:
        OLS_model = sm.OLS(dependent_variable, sm.add_constant(pd.concat([treatment_variable, propensity_score], axis = 1))).fit()
    else:
        OLS_model = sm.OLS(dependent_variable, sm.add_constant(pd.concat([treatment_variable, propensity_score], axis = 1))).fit(cov_type = cov_type)

    # Output the final result. The coefficient of the treatment variable is the target ATE estimation
    if output_tables is True:
        print(OLS_model.summary())

    # Return evaluation metric if needed
    if target_type == "neg_pvalue":
        return -OLS_model.pvalues[treatment_variable.name]
    elif target_type == "rsquared":
        return OLS_model.rsquared_adj
    elif target_type == "final_model":
        return OLS_model
    
@register_tool(tags=["econometric algorithm"])
def propensity_score_visualize_propensity_score_distribution(treatment_variable, propensity_score):
    
    '''
    VISUALIZE propensity score distribution for treatment group and control group and compare their distributions.
    The ideal result is that treatment group and control group should distribute similarly across propensity score. One common scenario is
    treatment group has most sample with propensity score close to 1 and control group has most sample with propensity score close to 0.
    In this scenario, the best solution is to trim samples with extreme propensity scores and obtain a subsample with similarly distributed propensity score.
    
    Args:
        treatment_variable (pd.Series): Target treatment variable, which should be a binary variable (1 for treatment, 0 for control).
        propensity_score (pd.Series): Propensity score for each sample to receive treatment, which should not contain nan value.
    '''
    
    # Obtain treatment group propensity score and control group propensity score respectively
    treatment_group_propensity = propensity_score.loc[treatment_variable[treatment_variable == 1].index]
    control_group_propensity = propensity_score.loc[treatment_variable[treatment_variable == 0].index]
    
    # Visualize the distribution using histogram
    plt.hist(control_group_propensity, bins = 40, facecolor = "blue", edgecolor = "black", alpha = 0.7, label = "control")
    plt.hist(treatment_group_propensity, bins = 40, facecolor = "red", edgecolor = "black", alpha = 0.7, label = "treatment")

#%%

@register_tool(tags=["econometric algorithm"])
def IV_2SLS_regression(dependent_variable, treatment_variable, IV_variable, covariate_variables, cov_type = None, target_type = None, output_tables = False):
    
    """
    Use Instrument Variable - Two Step Least Square (IV-2SLS) method to estimate Average Treatment Effect (ATE) of 
    the treatment variable towards the dependent variable, while ruling out endogeneiry in the original model.
    NOTE THAT THIS FUNCTION DOES NOT RETURN THE FINAL REGRESSION TABLE! All tables can (and only can) be printed out during the function.
    The final return is some clearly specified parameter or statistic within the regressions.

    Args:
        dependent_variable (pd.Series): Target dependent variable, which should not contain nan value.
        treatment_variable (pd.Series): Target treatment variable, which should not contain nan value.
        IV_variable (pd.Series or pd.DataFrame): Proposed instrument variable(s). Could have only one or multiple IVs. Should not contain nan value.
        covariate_variables (pd.DataFrame or None): Proposed covariate variables. If user does not specify any covariate variable, this could be None. Otherwise, it should not contain nan value.
        cov_type (str or None): The covariance estimator used in the results. If not specified by user, this could be None.
        target_type (str or None): Denote whether this function need to return any specific evaluation metric or any other content. If only want to output regression tables, this should be None. Also three possible inputs, "neg_pvalue" for the regression treatment variable coefficient p-value's negative value, "rsquared" for the adjusted R-squared value of the regression, and "final_model" for the final regression model.
        output_tables (bool): Denote whether this function need to print out regression tables. If want to see the tabels, this should be True. If only want the evaluation metric outputs, this should be False.
    """
    
    # First step regression
    if covariate_variables is None:
        first_step_X = IV_variable
    else:
        first_step_X = pd.concat([IV_variable, covariate_variables], axis = 1)
    if cov_type is None:
        first_step_regression = sm.OLS(treatment_variable, sm.add_constant(first_step_X)).fit()
    else:
        first_step_regression = sm.OLS(treatment_variable, sm.add_constant(first_step_X)).fit(cov_type = cov_type)
    predicted_treatment_result = pd.Series(first_step_regression.predict(sm.add_constant(first_step_X)), index = treatment_variable.index)
    predicted_treatment_result.name = treatment_variable.name + "_hat"

    # Second step regression
    if covariate_variables is None:
        second_step_X = predicted_treatment_result
    else:
        second_step_X = pd.concat([predicted_treatment_result, covariate_variables], axis = 1)
    if cov_type is None:
        second_step_regression = sm.OLS(dependent_variable, sm.add_constant(second_step_X)).fit()
    else:
        second_step_regression = sm.OLS(dependent_variable, sm.add_constant(second_step_X)).fit(cov_type = cov_type)

    # Output the final result. ATE is the coefficient of the predicted treatment variable
    if output_tables is True:
        print(second_step_regression.summary())

    # Return evaluation metric if needed
    if target_type == "neg_pvalue":
        return -second_step_regression.pvalues[predicted_treatment_result.name]
    elif target_type == "rsquared":
        return second_step_regression.rsquared_adj
    elif target_type == "final_model":
        return second_step_regression
    
@register_tool(tags=["econometric algorithm"])
def IV_2SLS_IV_setting_test(dependent_variable, treatment_variable, IV_variable, covariate_variables, cov_type = None):

    """
    Test the fundamental assuptions that the proposed Instrument Variable should satisfy, which are:
        1. Relevant Condition: In the proposed population model, the proposed IV should be relevant with the target treatment variable;
        2. Exclusion Restriction: In the proposed population model, the proposed IV should not be relevant with the residual of this model.

    Args:
        dependent_variable (pd.Series): Target dependent variable, which should not contain nan value.
        treatment_variable (pd.Series): Target treatment variable, which should not contain nan value.
        IV_variable (pd.Series): Proposed instrument variable. Could have ONLY ONE IV in this test function. Should not contain nan value.
        covariate_variables (pd.DataFrame or None): Proposed covariate variables. If user does not specify any covariate variable, this could be None. Otherwise, it should not contain nan value.
        cov_type (str or None): The covariance estimator used in the results. If not specified by user, this could be None.
    """

    # First test Relevant Condition. The coefficient of IV should be significant if it passes Relevant Condition requirement.
    if cov_type is None:
        relevant_test_OLS = sm.OLS(treatment_variable, sm.add_constant(IV_variable)).fit()
    else:
        relevant_test_OLS = sm.OLS(treatment_variable, sm.add_constant(IV_variable)).fit(cov_type = cov_type)
    print("Relevant Condition Test Result:")
    print(relevant_test_OLS.summary())
    
    # First test Exclusion Restriction. The coefficient of IV should be insignificant if it passes Exclusion Restriction requirement.
    if covariate_variables is None:
        restriction_test_X = treatment_variable
    else:
        restriction_test_X = pd.concat([treatment_variable, covariate_variables], axis = 1)
    if cov_type is None:
        restriction_test_OLS = sm.OLS(dependent_variable, sm.add_constant(restriction_test_X)).fit()
    else:
        restriction_test_OLS = sm.OLS(dependent_variable, sm.add_constant(restriction_test_X)).fit(cov_type = cov_type)
    residual_series = pd.Series(restriction_test_OLS.resid, index = restriction_test_X.index)
    if cov_type is None:
        restriction_test_final_OLS = sm.OLS(residual_series, sm.add_constant(IV_variable)).fit()
    else:
        restriction_test_final_OLS = sm.OLS(residual_series, sm.add_constant(IV_variable)).fit(cov_type = cov_type)        
    print("Exclusion Restriction Test Result:")
    print(restriction_test_final_OLS.summary())
    
#%%

@register_tool(tags=["econometric algorithm"])
def Static_Diff_in_Diff_regression(dependent_variable, 
                                   treatment_entity_dummy, 
                                   treatment_finished_dummy, 
                                   covariate_variables, 
                                   entity_effect = False, 
                                   time_effect = False, 
                                   other_effect = None, 
                                   cov_type = "unadjusted", 
                                   target_type = None, 
                                   output_tables = False):
    
    """
    Use Difference-in-Difference Regression method to estimate Average Treatment Effect (ATE) of 
    the treatment variable towards the dependent variable, in the PANEL DATA format. This is the STATIC version, 
    denoting that there is only one time spot when all entities in the treatment group is being treated. In other word, it's not the staggered method.
    NOTE THAT THIS FUNCTION DOES NOT RETURN THE FINAL REGRESSION TABLE! All tables can (and only can) be printed out during the function.
    The final return is some clearly specified parameter or statistic within the regressions.
    
    Args:
        dependent_variable (pd.Series): Target dependent variable, which should not contain nan value. The index of the series should be entity-time multi-index.
        treatment_entity_dummy (pd.Series): A dummy variables series denoting whether the entity is in the treatment group. This input should not contain nan value. The index of the series should be entity-time multi-index.
        treatment_finished_dummy (pd.Series): A dummy variables series denoting whether the treatment HAS BEEN implemented towards the treatment group. This input should not contain nan value. The index of the series should be entity-time multi-index.
        covariate_variables (pd.DataFrame or None): Proposed covariate variables. If user does not specify any covariate variable, this could be None. Otherwise, it should not contain nan value.
        entity_effect (bool): Denote whether entity effect is included in the regression.
        time_effect (bool): Denote whether time effect is included in the regression.
        other_effect (pd.DataFrame or None): Denote whether other effects are included in the regression. If there are other effects required, this input should be a pd.DataFrame with the categorial variable column(s) and entity-time multi-index. If no other effects required, leave this input to be None.
        cov_type (str): The covariance estimator used in the results. Five covariance estimators are supported: "unadjusted" for homoskedastic residual, "robust" for heteroskedasticity control, "cluster_entity" for entity clustering, "cluster_time" for time clustering, and "cluster_both" for entity-time two-way clustering.
        target_type (str or None): Denote whether this function need to return any specific evaluation metric or any other content. If only want to output regression tables, this should be None. Also three possible inputs, "neg_pvalue" for the regression treatment variable coefficient p-value's negative value, "rsquared" for the adjusted R-squared value of the regression, and "final_model" for the final regression model.
        output_tables (bool): Denote whether this function need to print out regression tables. If want to see the tabels, this should be True. If only want the evaluation metric outputs, this should be False.
    """
    
    # Check if inputs are proper formatted
    if cov_type not in ["unadjusted", "robust", "cluster_entity", "cluster_time", "cluster_both"]:
        raise RuntimeError("Covariance type input unsupported! This function supports 'unadjusted', 'robust', 'cluster_entity', 'cluster_time' and 'cluster_both' as possible inputs!")
    count_effects = 0
    if entity_effect is True:
        count_effects += 1
    if time_effect is True:
        count_effects += 1
    if other_effect is not None:
        count_effects += other_effect.shape[1]
    if count_effects > 2:
        raise RuntimeError("At most two effects allowed! Please note that now there are " + str(count_effects) + " effects in total!")

    # Check to ensure dummy variables
    if list(treatment_entity_dummy.map(int).sort_values().unique()) != [0, 1]:
        raise RuntimeError("treatment_entity_dummy Input Error! Please Check!")
    if list(treatment_finished_dummy.map(int).sort_values().unique()) != [0, 1]:
        raise RuntimeError("treatment_finished_dummy Input Error! Please Check!")
        
    # Prepare the dataset
    treatment_entity_dummy.name = "treatment_group"
    treatment_finished_dummy.name = "treated"
    beta = treatment_entity_dummy * treatment_finished_dummy
    beta.name = "treatment_group_treated"
    if covariate_variables is None:
        X = pd.concat([beta, treatment_entity_dummy, treatment_finished_dummy], axis = 1)
    else:
        X = pd.concat([beta, treatment_entity_dummy, treatment_finished_dummy, covariate_variables], axis = 1)
    if count_effects == 0:
        X = sm.add_constant(X)
    
    # Run the regression
    if cov_type in ["unadjusted", "robust"]:
        regression = PanelOLS(dependent_variable, X, entity_effect = entity_effect, time_effect = time_effect, other_effect = other_effect).fit(cov_type = cov_type)
    elif cov_type == "cluster_entity":
        regression = PanelOLS(dependent_variable, X, entity_effect = entity_effect, time_effect = time_effect, other_effect = other_effect).fit(cov_type = "clustered", cluster_entity = True)
    elif cov_type == "cluster_time":
        regression = PanelOLS(dependent_variable, X, entity_effect = entity_effect, time_effect = time_effect, other_effect = other_effect).fit(cov_type = "clustered", cluster_time = True)
    elif cov_type == "cluster_both":
        regression = PanelOLS(dependent_variable, X, entity_effect = entity_effect, time_effect = time_effect, other_effect = other_effect).fit(cov_type = "clustered", cluster_entity = True, cluster_time = True)

    # Output the final result. ATE is the coefficient of the predicted treatment variable
    if output_tables is True:
        print(regression)

    # Return evaluation metric if needed
    if target_type == "neg_pvalue":
        return -regression.pvalues[beta.name]
    elif target_type == "rsquared":
        return regression.rsquared
    elif target_type == "final_model":
        return regression
    
@register_tool(tags=["econometric algorithm"])
def Static_Diff_in_Diff_Parallel_Trend_visualization():
    pass  # TODO
    
@register_tool(tags=["econometric algorithm"])
def Staggered_Diff_in_Diff_regression(dependent_variable, 
                                      entity_treatment_dummy, 
                                      covariate_variables, 
                                      entity_effect = True, 
                                      time_effect = True, 
                                      other_effect = None, 
                                      cov_type = "unadjusted", 
                                      target_type = None, 
                                      output_tables = False):
    
    """
    Use Difference-in-Difference Regression method to estimate Average Treatment Effect (ATE) of 
    the treatment variable towards the dependent variable, in the PANEL DATA format. This is the STAGGERED version, 
    denoting that there could be multiple time spot when different entities in the treatment group are being gradually treated. In other word, it's not the static method.
    NOTE THAT THIS FUNCTION DOES NOT RETURN THE FINAL REGRESSION TABLE! All tables can (and only can) be printed out during the function.
    The final return is some clearly specified parameter or statistic within the regressions.
    
    Args:
        dependent_variable (pd.Series): Target dependent variable, which should not contain nan value. The index of the series should be entity-time multi-index.
        entity_treatment_dummy (pd.Series): A dummy variables series denoting whether the treatment HAS BEEN implemented towards the entity in the timeslot. This input should not contain nan value. The index of the series should be entity-time multi-index.
        covariate_variables (pd.DataFrame or None): Proposed covariate variables. If user does not specify any covariate variable, this could be None. Otherwise, it should not contain nan value.
        entity_effect (bool): Denote whether entity effect is included in the regression.
        time_effect (bool): Denote whether time effect is included in the regression.
        other_effect (pd.DataFrame or None): Denote whether other effects are included in the regression. If there are other effects required, this input should be a pd.DataFrame with the categorial variable column(s) and entity-time multi-index. If no other effects required, leave this input to be None.
        cov_type (str): The covariance estimator used in the results. Five covariance estimators are supported: "unadjusted" for homoskedastic residual, "robust" for heteroskedasticity control, "cluster_entity" for entity clustering, "cluster_time" for time clustering, and "cluster_both" for entity-time two-way clustering.
        target_type (str or None): Denote whether this function need to return any specific evaluation metric or any other content. If only want to output regression tables, this should be None. Also three possible inputs, "neg_pvalue" for the regression treatment variable coefficient p-value's negative value, "rsquared" for the adjusted R-squared value of the regression, and "final_model" for the final regression model.
        output_tables (bool): Denote whether this function need to print out regression tables. If want to see the tabels, this should be True. If only want the evaluation metric outputs, this should be False.
    """
    
    # Check if inputs are proper formatted
    if cov_type not in ["unadjusted", "robust", "cluster_entity", "cluster_time", "cluster_both"]:
        raise RuntimeError("Covariance type input unsupported! This function supports 'unadjusted', 'robust', 'cluster_entity', 'cluster_time' and 'cluster_both' as possible inputs!")
    count_effects = 0
    if entity_effect is True:
        count_effects += 1
    if time_effect is True:
        count_effects += 1
    if other_effect is not None:
        count_effects += other_effect.shape[1]
    if count_effects > 2:
        raise RuntimeError("At most two effects allowed! Please note that now there are " + str(count_effects) + " effects in total!")

    # Check to ensure dummy variables
    if list(entity_treatment_dummy.map(int).sort_values().unique()) != [0, 1]:
        raise RuntimeError("entity_treatment_dummy Input Error! Please Check!")
        
    # Prepare the dataset
    entity_treatment_dummy.name = "treatment_entity_treated"
    if covariate_variables is None:
        X = entity_treatment_dummy
    else:
        X = pd.concat([entity_treatment_dummy, covariate_variables], axis = 1)
    if count_effects == 0:
        X = sm.add_constant(X)
    
    # Run the regression
    if cov_type in ["unadjusted", "robust"]:
        regression = PanelOLS(dependent_variable, X, entity_effect = entity_effect, time_effect = time_effect, other_effect = other_effect).fit(cov_type = cov_type)
    elif cov_type == "cluster_entity":
        regression = PanelOLS(dependent_variable, X, entity_effect = entity_effect, time_effect = time_effect, other_effect = other_effect).fit(cov_type = "clustered", cluster_entity = True)
    elif cov_type == "cluster_time":
        regression = PanelOLS(dependent_variable, X, entity_effect = entity_effect, time_effect = time_effect, other_effect = other_effect).fit(cov_type = "clustered", cluster_time = True)
    elif cov_type == "cluster_both":
        regression = PanelOLS(dependent_variable, X, entity_effect = entity_effect, time_effect = time_effect, other_effect = other_effect).fit(cov_type = "clustered", cluster_entity = True, cluster_time = True)

    # Output the final result. ATE is the coefficient of the predicted treatment variable
    if output_tables is True:
        print(regression)

    # Return evaluation metric if needed
    if target_type == "neg_pvalue":
        return -regression.pvalues[entity_treatment_dummy.name]
    elif target_type == "rsquared":
        return regression.rsquared
    elif target_type == "final_model":
        return regression

@register_tool(tags=["econometric algorithm"])
def Staggered_Diff_in_Diff_Event_Study_regression(dependent_variable, 
                                                  entity_treatment_dummy, 
                                                  covariate_variables, 
                                                  see_back_length: int = 4, 
                                                  see_forward_length: int = 3, 
                                                  entity_effect = True, 
                                                  time_effect = True, 
                                                  other_effect = None, 
                                                  cov_type = "unadjusted", 
                                                  target_type = None, 
                                                  output_tables = False):

    """
    Use Difference-in-Difference Regression method to estimate Average Treatment Effect (ATE) of 
    the treatment variable towards the dependent variable, in the PANEL DATA format. This is the STAGGERED version, 
    denoting that there could be multiple time spot when different entities in the treatment group are being gradually treated. In other word, it's not the static method.
    Also, this is the event study version, denoting that there should be enough amount of different treatment implementation time spots (more than see_back_length + see_forward_length).
    NOTE THAT THIS FUNCTION DOES NOT RETURN THE FINAL REGRESSION TABLE! All tables can (and only can) be printed out during the function.
    The final return is some clearly specified parameter or statistic within the regressions.
    
    Args:
        dependent_variable (pd.Series): Target dependent variable, which should not contain nan value. The index of the series should be entity-time multi-index.
        entity_treatment_dummy (pd.Series): A dummy variables series denoting whether the treatment HAS BEEN implemented towards the entity in the timeslot. This input should not contain nan value. The index of the series should be entity-time multi-index.
        covariate_variables (pd.DataFrame or None): Proposed covariate variables. If user does not specify any covariate variable, this could be None. Otherwise, it should not contain nan value.
        entity_effect (bool): Denote whether entity effect is included in the regression.
        see_back_length (int): A positive int denote the length of see-back observation. 
        see_forward_length (int): A positive int denote the length of see-forward observation. 
        time_effect (bool): Denote whether time effect is included in the regression.
        other_effect (pd.DataFrame or None): Denote whether other effects are included in the regression. If there are other effects required, this input should be a pd.DataFrame with the categorial variable column(s) and entity-time multi-index. If no other effects required, leave this input to be None.
        cov_type (str): The covariance estimator used in the results. Five covariance estimators are supported: "unadjusted" for homoskedastic residual, "robust" for heteroskedasticity control, "cluster_entity" for entity clustering, "cluster_time" for time clustering, and "cluster_both" for entity-time two-way clustering.
        target_type (str or None): Denote whether this function need to return any specific evaluation metric or any other content. If only want to output regression tables, this should be None. Also three possible inputs, "neg_pvalue" for the regression treatment variable coefficient p-value's negative value, "rsquared" for the adjusted R-squared value of the regression, and "final_model" for the final regression model.
        output_tables (bool): Denote whether this function need to print out regression tables. If want to see the tabels, this should be True. If only want the evaluation metric outputs, this should be False.
    """
    
    # Check if inputs are proper formatted
    if cov_type not in ["unadjusted", "robust", "cluster_entity", "cluster_time", "cluster_both"]:
        raise RuntimeError("Covariance type input unsupported! This function supports 'unadjusted', 'robust', 'cluster_entity', 'cluster_time' and 'cluster_both' as possible inputs!")
    count_effects = 0
    if entity_effect is True:
        count_effects += 1
    if time_effect is True:
        count_effects += 1
    if other_effect is not None:
        count_effects += other_effect.shape[1]
    if count_effects > 2:
        raise RuntimeError("At most two effects allowed! Please note that now there are " + str(count_effects) + " effects in total!")

    # Check to ensure dummy variables are proper formatted
    if list(entity_treatment_dummy.map(int).sort_values().unique()) != [0, 1]:
        raise RuntimeError("entity_treatment_dummy Input Error! Please Check!")
        
    # =========================================================================
        
    # Construct the event study variables
    entity_index_name, time_index_name = entity_treatment_dummy.index.names[0], entity_treatment_dummy.index.names[1]
    treatment_name = entity_treatment_dummy.name
    data_df = entity_treatment_dummy.reset_index()
    all_entity_list = list(data_df.entity_index_name.unique())
    all_time_list = list(data_df.time_index_name.unique())
    all_time_list.sort()
    policy_implementation_time_record_list = []
    for each_entity in all_entity_list:
        temp_df = data_df[data_df[entity_index_name] == each_entity]
        temp_df = temp_df.sort_values(by = time_index_name)
        if temp_df[temp_df[treatment_name] == 0].shape[0] == 0:
            raise RuntimeError("Entity:", each_entity, "was keeping implementing the policy since the start of this sample! Should consider to drop this entity!")
            # continue
        check_series = temp_df[treatment_name] - temp_df[treatment_name].shift().fillna(0)
        if check_series[check_series == 1].shape[0] == 0:
            # print("Entity:", each_entity, "did not implement this policy!")
            continue
        policy_time = temp_df.loc[check_series[check_series == 1].index[0], time_index_name]
        # print("Entity:", each_entity, "implemented this policy in time:", policy_time)
        if policy_time not in policy_implementation_time_record_list:
            policy_implementation_time_record_list.append(policy_time)
        
    # Check to ensure event study settings are proper formatted
    if see_back_length < 4 or see_forward_length < 3:
        raise RuntimeError("See back day length or see forward day length too few! Please check!")
    elif see_back_length + see_forward_length >= len(all_time_list):
        raise RuntimeError("See back day length or see forward day length too large! Please check!")
    
    # Construct Lead-Lag Dummy Variables (set Lead_D1 as default)
    Lead_column_name_list = ["Lead_D" + str(see_back_length) + "+"]
    for i in np.arange(see_back_length - 1, 1, -1):
        Lead_column_name_list.append("Lead_D" + str(i))
    Lag_column_name_list = []
    for i in np.arange(1, see_forward_length, 1):
        Lag_column_name_list.append("Lag_D" + str(i))
    Lag_column_name_list.append("Lag_D" + str(see_forward_length) + "+")
    Lead_and_Lag_column_name_list = Lead_column_name_list + ["D0"] + Lag_column_name_list
    
    # Calculate the values for each dummy variable
    considered_data_df = data_df[["entity", "time"]]
    considered_data_df[Lead_and_Lag_column_name_list] = np.nan
    for each_entity in all_entity_list:
        temp_df = data_df[data_df[entity_index_name] == each_entity]
        check_series = temp_df[treatment_name] - temp_df[treatment_name].shift().fillna(0)
        if check_series[check_series == 1].shape[0] == 0:  # If the state never implement the policy
            considered_data_df.loc[considered_data_df["entity"] == each_entity, Lead_and_Lag_column_name_list] = 0
            continue
        # If the state always implement the policy, for this version such case should already be removed. So do not consider it here.
        policy_time_index = check_series[check_series == 1].index[0]
        for each_index in temp_df.index:
            corresponding_each_time = temp_df.loc[each_index, time_index_name]
            if each_index - policy_time_index <= -see_back_length:
                considered_data_df.loc[(considered_data_df["entity"] == each_entity) & (considered_data_df["time"] == corresponding_each_time), "Lead_D" + str(see_back_length) + "+"] = 1
            elif each_index - policy_time_index > -see_back_length and each_index - policy_time_index < -1:
                considered_data_df.loc[(considered_data_df["entity"] == each_entity) & (considered_data_df["time"] == corresponding_each_time), "Lead_D" + str(policy_time_index - each_index)] = 1
            elif each_index == policy_time_index:
                considered_data_df.loc[(considered_data_df["entity"] == each_entity) & (considered_data_df["time"] == corresponding_each_time), "D0"] = 1
            elif each_index - policy_time_index > 0 and each_index - policy_time_index < see_forward_length:
                considered_data_df.loc[(considered_data_df["entity"] == each_entity) & (considered_data_df["time"] == corresponding_each_time), "Lag_D" + str(each_index - policy_time_index)] = 1
            elif each_index - policy_time_index >= see_forward_length:
                considered_data_df.loc[(considered_data_df["entity"] == each_entity) & (considered_data_df["time"] == corresponding_each_time), "Lag_D" + str(see_forward_length) + "+"] = 1
        considered_data_df.loc[considered_data_df["entity"] == each_entity, Lead_and_Lag_column_name_list] = considered_data_df.loc[considered_data_df["entity"] == each_entity, Lead_and_Lag_column_name_list].fillna(0)
    considered_data_df = considered_data_df.set_index(["entity", "time"])
    
    # =========================================================================

    # Prepare the dataset
    if covariate_variables is None:
        X = considered_data_df
    else:
        X = pd.concat([considered_data_df, covariate_variables], axis = 1)

    # Run the regression
    if cov_type in ["unadjusted", "robust"]:
        regression = PanelOLS(dependent_variable, X, entity_effect = entity_effect, time_effect = time_effect, other_effect = other_effect).fit(cov_type = cov_type)
    elif cov_type == "cluster_entity":
        regression = PanelOLS(dependent_variable, X, entity_effect = entity_effect, time_effect = time_effect, other_effect = other_effect).fit(cov_type = "clustered", cluster_entity = True)
    elif cov_type == "cluster_time":
        regression = PanelOLS(dependent_variable, X, entity_effect = entity_effect, time_effect = time_effect, other_effect = other_effect).fit(cov_type = "clustered", cluster_time = True)
    elif cov_type == "cluster_both":
        regression = PanelOLS(dependent_variable, X, entity_effect = entity_effect, time_effect = time_effect, other_effect = other_effect).fit(cov_type = "clustered", cluster_entity = True, cluster_time = True)

    # Output the final result. ATE is the coefficient of the predicted treatment variable
    if output_tables is True:
        print(regression)

    # Return evaluation metric if needed
    if target_type == "neg_pvalue":
        return -regression.pvalues["D0"]
    elif target_type == "rsquared":
        return regression.rsquared
    elif target_type == "final_model":
        return regression
    
# @register_tool(tags=["econometric algorithm"])
# def Staggered_Diff_in_Diff_Event_Study_visualization():
#     pass  # TODO
    
# #%%

# @register_tool(tags=["econometric algorithm"])
# def Regression_Discontinuity_Design_regression():
#     pass  # TODO

# @register_tool(tags=["econometric algorithm"])
# def Regression_Discontinuity_Design_visualization():
#     pass  # TODO

# #%%
# #%%
# #%%
# #%%
# #%%
# #%%

# @register_tool(tags=["econometric algorithm"])
# def Synthetic_Control_construction():
#     pass  # TODO

# @register_tool(tags=["econometric algorithm"])
# def Synthetic_Control_Diff_in_Diff_regression():
#     pass  # TODO

# @register_tool(tags=["econometric algorithm"])
# def Synthetic_Control_Event_Study_visualization():
#     pass  # TODO

# @register_tool(tags=["econometric algorithm"])
# def Synthetic_Control_Regression_Discontinuity_Design_regression():
#     pass  # TODO

# @register_tool(tags=["econometric algorithm"])
# def Synthetic_Control_Regression_Discontinuity_Design_visualization():
#     pass  # TODO

# #%%

# @register_tool(tags=["econometric algorithm"])
# def Propensity_Score_Matching_construction():
#     pass  # TODO

# @register_tool(tags=["econometric algorithm"])
# def Propensity_Score_Matching_Diff_in_Diff_regression():
#     pass  # TODO

# @register_tool(tags=["econometric algorithm"])
# def Propensity_Score_Matching_Regression_Discontinuity_Design_regression():
#     pass  # TODO