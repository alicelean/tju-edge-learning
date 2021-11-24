#定义一些变量记录程序运行的结果
import pandas as pd
import os
#定义server和client之间交互的次数
global COM_TIMES
COM_TIMES=1
global DF
DF=pd.DataFrame(columns=['COM_TIMES','tau','time_total_all','local_time','time_global_aggregation_all','loss'])
# PATH="/Users/alice/tju.com/python/tju-edge-learning/result_value/flie/"
# print(PATH)
PATH=os.path.dirname(__file__)+"/flie/"