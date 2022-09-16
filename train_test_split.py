# 按顺序抽取训练集和测试集

import pandas as pd
# 读取csv文件
df = pd.read_csv("simplifyweibo_4_moods.csv")
# 获取文件总行数
row_num = len(df)
print(row_num)
stop = int(row_num*0.8)
# # 确定每个小文件要包含的数量
# step = 400
d1 = df[0:stop]
print(len(d1))
d1.to_csv("1.csv", index=None)
d2 = df[stop:row_num]
d2.to_csv("2.csv", index=None)
print(len(d2))




# 随机打乱抽取出训练集和测试集

df = pd.read_csv('simplifyweibo_4_moods.csv', encoding='utf-8')

# df.drop_duplicates(keep='first', inplace=True)  # 去重，只保留第一次出现的样本

df = df.sample(frac=1.0)  # 全部打乱
#  0.2 为比例值
cut_idx = int(round(0.2 * df.shape[0]))

df_test, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]
df_train.to_csv("weibo_3_moods_train.csv", index=None)
df_test.to_csv("weibo_3_moods_test.csv", index=None)
print (df.shape, df_test.shape, df_train.shape)  # (3184, 12) (318, 12) (2866, 12)