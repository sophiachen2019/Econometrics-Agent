import asyncio

from metagpt.actions import WriteAnalysisCode
from metagpt.roles.di.data_interpreter import DataInterpreter
from metagpt.utils.recovery_util import save_history
from shared_queue import log_execution


# async def main(requirement: str):
#     role = DataInterpreter(use_reflection=True, tools=["<all>"])
#     # role = DataInterpreter(use_reflection=True)
#     await role.run(requirement)
#     save_history(role=role)



async def main_generator_with_interpreter(interpreter: DataInterpreter, requirement: str, user_id: str):
    await log_execution("#### 🔥Starting main function\n", user_id)
    role = interpreter  # 假设 'interpreter' 类似于 'role'
    role.set_actions([WriteAnalysisCode])
    role._set_state(0)
    await role.run(requirement, user_id)
    save_history(role=role)
    await log_execution("#### Finished main function😊\n", user_id)

async def main_generator(requirement1: str):
    await log_execution("#### 🔥Starting main function\n", "1")
    # 创建两个DataInterpreter实例
    role1 = DataInterpreter(use_reflection=True, tools=["<all>"])
    # role2 = DataInterpreter(use_reflection=True, tools=["<all>"])
    # 同时运行两个实例
    await asyncio.gather(
        role1.run(requirement1, user_id="1"),
        # role2.run(requirement2)
    )
    
    # 设置两个实例的actions
    # role1.set_actions([WriteAnalysisCode])
    # role2.set_actions([WriteAnalysisCode])
    
    # # 重置状态
    # role1._set_state(0)
    # role2._set_state(0)
    
    # 保存历史记录
    save_history(role=role1)
    # save_history(role=role2)
    
    await log_execution("#### Finished main function😊\n", "1")


if __name__ == "__main__":
    # data_path = "your/path/to/titanic"
    # train_path = f"{data_path}/split_train.csv"
    # eval_path = f"{data_path}/split_eval.csv"
    # requirement = f"This is a titanic passenger survival dataset, your goal is to predict passenger survival outcome. The target column is Survived. Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. Report accuracy on the eval data. Train data path: '{train_path}', eval data path: '{eval_path}'."
    # requirement = "Plot a heatmap of the global warming trends for each of the European Union countries using the dataset at https://opendata.arcgis.com/datasets/4063314923d74187be9596f10d034914_0.csv . Save the graph at the current working dictionary named 'heatmap'. Put countries on the y-axis and years on the x-axis."
    requirement1 = "请帮我写一份冒泡排序算法，并帮我测试其是否是可行的"
    # requirement2 = "please use the previous result add 2"
    
    # requirement = "This is a house price dataset, your goal is to predict the sale price of a property based on its features. The target column is SalePrice. Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. Report RMSE between the logarithm of the predicted value and the logarithm of the observed sales price on the eval data. Train data path: '/Users/tuozhou/Desktop/RA/SZRI/ML_Assistant/data/05_house-prices-advanced-regression-techniques/split_train.csv', eval data path: '/Users/tuozhou/Desktop/RA/SZRI/ML_Assistant/data/05_house-prices-advanced-regression-techniques/split_eval.csv'."
    asyncio.run(main_generator(requirement1))
