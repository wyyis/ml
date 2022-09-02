import jieba
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
def datasets_demo():
    """
    数据集的使用
    鸢尾花
    :return:
    """
    # 1、获取鸢尾花数据集
    iris = load_iris()
    print("鸢尾花数据集的返回值：\n", iris)
    # 返回值是一个继承自字典的Bench
    print("鸢尾花的特征值:\n", iris["data"])
    print("鸢尾花的目标值：\n", iris.target)
    print("鸢尾花特征的名字：\n", iris.feature_names)
    print("鸢尾花目标值的名字：\n", iris.target_names)
    print("鸢尾花的描述：\n", iris.DESCR)

    # 2、对鸢尾花数据集进行分割
    # 训练集的特征值x_train 测试集的特征值x_test 训练集的目标值y_train 测试集的目标值y_test
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)
    print("x_train:\n", x_train.shape)
    # 随机数种子
    x_train1, x_test1, y_train1, y_test1 = train_test_split(iris.data, iris.target, random_state=6)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(iris.data, iris.target, random_state=6)
    print("如果随机数种子不一致：\n", x_train == x_train1)
    print("如果随机数种子一致：\n", x_train1 == x_train2)

    return None
def dic_demo2():
    """
    进行字典特征提取
    :return:
    """
    data = [{'city': '北京', 'temperature': 200}, {'city': '上海', 'temperature': 100}, {'city': '深圳', 'temperature': 90}]
    transfer = DictVectorizer(sparse=False)
    data_new = transfer.fit_transform(data)
    print(data_new)
    print(transfer.get_feature_names())


def count_en_demo2():
    """
    进行英文文本特征提取
    :return:
    """
    data = ["life is short, i like like python", "life is long,i dislike python"]
    transfer = CountVectorizer(stop_words=["is"])
    data_new = transfer.fit_transform(data)
    print(data_new.toarray())
    print(transfer.get_feature_names())


def cut_word(text):
    """
    进行中文分词
    :param text:
    :return:
    """
    return " ".join(jieba.cut(text))


def count_chinese_demo2():
    """
    进行中文文本特征提取
    :return:
    """
    data = ["中国诗歌文化博大精深，意韵悠长，", "本文选取了一些人们比较熟悉的诗词，尝试从不同时代，不同背景，", "领略到诗词的意境，从中获得艺术享受，表达自己的看法。"]
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    print(data_new)
    transfer = CountVectorizer()
    data_final = transfer.fit_transform(data_new)
    print(data_final.toarray(), data_final.shape)
    print(transfer.get_feature_names())

def tfidf_demo():
    """
    用TF-IDF的方法进行文本特征抽取
    :return:
    """

    data = ["南京市长江大桥", "我爱加拿大文化", "我爱美国文化"]
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    print(data_new)
    transfer = TfidfVectorizer()
    data_final = transfer.fit_transform(data_new)
    print(data_final.toarray(), data_final.shape)
    print(transfer.get_feature_names())

def minmax_demo():
    """
    归一化
    :return:
    """
    #1、获取数据
    data = pd.read_csv("datingTestSet2.csv")
    data = data.iloc[:,0:3]
    print(data)


    #2、实例化转化器类
    transfer = MinMaxScaler(feature_range=[2,3])
    #3、调用fit_transform
    data_new = transfer.fit_transform(data)
    print(data_new)

def standard_demo():
    """
    标准化处理
    处理成期望为0，方差为1的一组数据
    对特征进行无量纲处理，一般用的是标准化
    :return:
    """
    data = pd.read_csv("datingTestSet2.csv")
    data = data.iloc[:, 0:3]
    transfer = StandardScaler()
    data_new = transfer.fit_transform(data)
    print(data_new)
def variance_demo():
    """
    过滤低方差特征
    :return:
    """
    data = [
        [1, 1, 3, 4, 5],
        [1, 1, 8, 9, 10],
        [1, 2, 13, 14, 15]
    ]
    transfer = VarianceThreshold(threshold=1)
    data_new = transfer.fit_transform(data)
    print(data_new)

    # 计算某两个变量之间的相关系数
    print(pearsonr([1,2,3],[4,5,6]))
def pca_demo():
    """
    主成分分析
    :return:
    """
    data = [[2,8,4,5],[6,3,0,8],[5,4,9,1]]
    # 参数为小数 : 保留多少的信息，如0.9表示保留90%的信息
    # 参数为整数 : 降低到多少维
    transfer = PCA(n_components=0.95)
    data_new = transfer.fit_transform(data)
    print(data_new)
if __name__ == "__main__":
    # count_chinese_demo2()
    standard_demo()