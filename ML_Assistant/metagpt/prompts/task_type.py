# Prompt for taking on "eda" tasks
EDA_PROMPT = """
The current task is about exploratory data analysis, please note the following:
- Distinguish column types with `select_dtypes` for tailored analysis and visualization, such as correlation.
- Remember to `import numpy as np` before using Numpy functions.
"""

# Prompt for taking on "data_preprocess" tasks
DATA_PREPROCESS_PROMPT = """
The current task is about data preprocessing, please note the following:
- Monitor data types per column, applying appropriate methods ONLY WHEN NECESSARY OR REQUIRED.
- Ensure operations are on existing dataset columns.
- Avoid writing processed data to files.
- Avoid any change to label column, such as standardization, etc.
- Prefer alternatives to one-hot encoding for categorical data.
- Only encode or scale necessary columns to allow for potential feature-specific engineering tasks (like time_extract, binning, extraction, etc.) later.
- IF there is train-test split operation, each step do data preprocessing to train, must do same for test separately at the same time.
- Always copy the DataFrame before processing it and use the copy to process.
"""

# Prompt for taking on "econometric_algorithm" tasks
ECONOMETRIC_ALGORITHM_PROMPT = """
The current task is about matching and applying an econometric algorithm tool. please note the following:
- ALWAYS APPLY AVAILABLE TOOLS FIRST. Find the tool that most satisfy the target.
- Only when no tools are proper for the target should you initiate to find a model. 
- Use the data from previous task result directly, do not mock or reload data yourself.
- Some results are not quantities and may contain graphs and/or tables. STRICTLY follow the user's instruction and provide possible qualitative conclusions and/or suggestions.
"""

# Prompt for taking on "econometric_optimization" tasks
ECONOMETRIC_OPTIMIZATION_PROMPT = """
The current task is about grid-searching the best data pre-processing method and\or hyper-parameter to optimize the final result. please note the following:
- ONLY START THIS TASK WHEN THE USER CLEARLY INSTRUCTS TO OPTIMIZE THE FINAL RESULT.
- There are different optimization tools for different econometric analysis methods. Find exactly the matched tool for the target econometric analysis method.
- The input data should not contain any nan value. MAKE SURE to deal with nan values and make datasets tidy (for example, dependent variable and treatment variable and other variables have their index well matched) before passed to the optimization tool.
- There might be data columns that are categorical or dummy variables but in default data type of float or int, but BE SURE TO CHECK AND FIGURE THEM OUT BEFORE THE OPTIMIZATION ALGORITHM STARTS! Many optimization algorithm requires to know which data columns are categorical or dummy variables.
- If the user specifies not to implement some type of optimization method, remember to adjust the parameter passed to the optimization tool.
"""