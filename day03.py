from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error
def linear1():
    """
    正规方程的优化方法，对Boston房价进行预测
    :return:
    """
    # 1) 获取数据
    boston = load_boston()
    # 2) 划分数据
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)
    # 3) 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    # 4) 预估器处理
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)
    # 5) 得出模型
    print("正规方程-权重系数为：", estimator.coef_)
    print("正规方程-偏置为：", estimator.intercept_)
    # 6) 模型评估
    print("正规方程-模型准确度为：", estimator.score(x_test, y_test))
    y_predict = estimator.predict(x_test)
    print("正规方程-预测房价：", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("正规方程-均方误差为：", error)

def linear2():
    """
    梯度下降的优化方法，对Boston房价进行预测
    :return:
    """
    # 1) 获取数据
    boston = load_boston()
    # 2) 划分数据
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)
    # 3) 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    # 4) 预估器处理
    estimator = SGDRegressor(learning_rate="constant", eta0=0.01, max_iter=10000, penalty="l1")
    estimator.fit(x_train, y_train)
    # 5) 得出模型
    print("梯度下降-权重系数为：", estimator.coef_)
    print("梯度下降-偏置为：", estimator.intercept_)
    # 6) 模型评估
    print("梯度下降-模型准确度为：", estimator.score(x_test, y_test))
    y_predict = estimator.predict(x_test)
    print("梯度下降-预测房价：", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("梯度下降-均方误差为：", error)

def linear3():
    """
    岭回归的优化方法，对Boston房价进行预测
    :return:
    """
    # 1) 获取数据
    boston = load_boston()
    # 2) 划分数据
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)
    # 3) 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    # 4) 预估器处理
    estimator = Ridge(max_iter=10000, alpha=0.5)
    estimator.fit(x_train, y_train)
    # 5) 得出模型
    print("岭回归-权重系数为：", estimator.coef_)
    print("岭回归-偏置为：", estimator.intercept_)
    # 6) 模型评估
    print("岭回归-模型准确度为：", estimator.score(x_test, y_test))
    y_predict = estimator.predict(x_test)
    print("岭回归-预测房价：", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("岭回归-均方误差为：", error)

    
if __name__ == "__main__":
    # 1.正规方程
    linear1()
    # 2.梯度下降
    linear2()
    # 3.岭回归
    linear3()
