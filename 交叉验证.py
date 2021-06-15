import numpy as np
import pandas as pd
import catboost as cbt
from sklearn.metric import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
 
 
train_data = pd.read_csv('.....')
train_label = pd.read_csv('.....')
test_data = pd.read_csv('.....')
 
 
cat_list = [] #catboost中需要处理的离散特征属性列
oof = np.zeros(train_data.shape[0])  #训练集长度
prediction = np.zeros(test_data.shape[0])     #测试集长度
seeds = [2017,2018,2019,2020]     #随机种子
num_model_seed = 1
 
 
train_x, test_x, train_y, test_y = train_test_split(train_data, train_label, test_size=0.2, random_state = 2019)     #拆分训练集
for model_seed in range(num_model_seed):     #选用几个随机种子
    oof_cat = np.zeros(train_data.shape[0])
    prediction_cat = np.zeros(test_data.shape[0])
    skf = StratifiedKFold(n_splits=5, random_state=seeds[model_seed], shuffle=True) #五折交叉验证，shuffle表示是否打乱数据，若为True再设定随机种子random_state
    for index, (train_index, test_index) in enumerate(skf.split(train_data, train_label)): #将数据五折分割
        train_x, test_x, train_y, test_y = train_data.iloc[train_index], test_data.iloc[test_index], train_label.iloc[train_index], train_label.iloc[test_index]
        cbt_model = cbt.CatBoostClassifier(iterations=5000, learning_rate=0.1, max_depth=7, verbose=100, early_stopping_rounds=500,task_type='GPU', eval_metric='F1',cat_features=cat_list)     #设置模型参数，verbose表示每100个训练输出打印一次
        cbt_model.fit(train_x, train_y, eval_set=(test_x, test_y)) #训练五折分割后的训练集
        gc.collect() #垃圾清理，内存清理
        oof_cat[test_index] += cbt_model.predict_proba(test_x)[:,1] #
        prediction_cat += cbt_model.predict_proba(test_data)[:,1]/5
    print('F1', f1_score(train_label, np.round(oof_cat)))
    oof += oof_cat / num_model_seed     #五折训练集取均值
    prediction += prediction_cat / num_model_seed #测试集取均值
print('score', f1_score(train_label, np.round(oof)))
 
##结果写入csv文件
submit = test_data['id']
submit['label'] = (prediction>=0.499).astype(int)
print(submit['label'].value_counts())
submit.to_csv('submission.csv',index=False)
