from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
def knn_iris():
    """
    用KNN算法对鸢尾花进行分类
    :return:
    """
    # 1、获取数据
    iris = load_iris()
    # 2、划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris["data"], iris["target"], random_state=6)
    # 3、特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 4、KNN算法预估器
    # k值一般不适用偶数
    estimator = KNeighborsClassifier(n_neighbors=11)
    estimator.fit(x_train, y_train)
    # 5、模型评估
    # (1)、直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print(y_test == y_predict)
    # (2)、计算准确率
    score = estimator.score(x_test, y_test)
    print(score)
    return None


def knn_iris_gscv():
    """
    用KNN算法对鸢尾花进行分类
    添加网格搜索和交叉验证
    :return:
    """
    # 1、获取数据
    iris = load_iris()
    # 2、划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris["data"], iris["target"], random_state=6)
    # 3、特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 4、KNN算法预估器
    # k值一般不适用偶数
    estimator = KNeighborsClassifier()

    # 参数准备
    param_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11]}
    # 加入网格搜索和交叉验证
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)

    estimator.fit(x_train, y_train)
    # 5、模型评估
    # (1)、直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print(y_test == y_predict)
    # (2)、计算准确率
    # 求的是测试集的准确率
    score = estimator.score(x_test, y_test)
    print(score)

    print(estimator.best_estimator_)
    print(estimator.best_params_)
    # 求的是交叉验证时验证集的最佳准确率
    print(estimator.best_score_)
    print(estimator.cv_results_)
    return None
def nb_news():
    """
    用朴素贝叶斯对新闻文本进行分类
    :return:
    """
    # 1) 获取数据
    news = fetch_20newsgroups(subset="all")
    # 2) 划分数据集
    x_train, x_test, y_train, y_test =  train_test_split(news.data, news.target)
    # 3) 特征工程：文本特征抽取-tfidf
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 4) 朴素贝叶斯算法预估器
    estimator = MultinomialNB(alpha=0.01)
    estimator.fit(x_train, y_train)
    # 5) 模型评估
    y_predict = estimator.predict(x_test)
    print(y_predict)
    print(y_test == y_predict)
    print(estimator.score(x_test, y_test))

def decision_iris():
    """
    用决策树对鸢尾花数据进行分类
    :return:
    """
    # 1）获取数据集
    iris = load_iris()
    # 2）划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=6)
    # 3）决策树预估器
    estimator = DecisionTreeClassifier(criterion='entropy')
    estimator.fit(x_train, y_train)
    # 4）模型评估
    print(estimator.score(x_test, y_test))
    # 5) 可视化决策树
    export_graphviz(estimator, out_file="iris_tree.dot ", feature_names=iris.feature_names)

if __name__ == "__main__":
    decision_iris()
