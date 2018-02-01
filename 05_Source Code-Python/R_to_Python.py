
# coding: utf-8

# # R转Python

# ## 6.2 统计分析

# ### （1）数据读入

# In[7]:


# 导入Python做数据处理的模块pandas，并取别名为pd
# 导入numpy模块，并取别名为np
# 从pandas模块中导入DataFrame和Series类
import pandas as pd
import numpy as np


# In[10]:


#设置当前工作目录
    #【注】“当前工作目录”的含义为文件和文件夹的读写路径
os.chdir('H:\PythonProjects')
print(os.getcwd())


# In[13]:


# 调用pandas的read_csv()函数，读取一个csv文件，并创建一个DataFrame
    # 注：women.csv源自R数据集
women = pd.read_csv('women.csv', index_col=0)
print(women.head())


# ### （2）数据理解

# In[15]:


# 查看描述性统计分析
women.describe()


# In[17]:


# 查看列名
print(women.columns)


# In[19]:


# 查看形状
print('行数：', women.shape[0])
print('列数：', women.shape[1])


# ### （3）数据建模

# In[65]:


# 从机器学习模块中导入线性回归类LinearRegression
from sklearn.linear_model import LinearRegression

# 构建模型训练集，由于数据较少，全部数据用于训练
# 并设置训练集中的自变量与因变量

# 选取特征变量值为women.height,构造特征矩阵
# 当特征变量为一个时，因调用reshape(-1, 1)方法用于构造特征矩阵
X_train = women.height.values.reshape(-1, 1)
X_train


# In[66]:


# 选取响应变量
y_train = women.weight


# In[69]:


# 实例化模型
    # fit_intercept参数用于设置是否训练截距
model = LinearRegression(fit_intercept=True)


# In[74]:


# 训练模型
model.fit(X_train, y_train)


# ### （4）查看模型

# In[77]:


# 查看模型的斜率
    # 训练模型的斜率为一个列表对象，依次为各自变量的斜率
print("训练模型斜率为：", model.coef_[0])


# In[78]:


# 查看模型的截距
print("训练模型截距为：", model.intercept_)


# ### （5）模型预测
# 

# In[79]:


# 用训练的模型预测对原体重数据进行预测
# 返回结果为numpy的数组类型

predicted_weight = model.predict(women.height.values.reshape(-1, 1))
print(predicted_weight)


# In[80]:


# 将原体重数据转换为数组，并查看其值
print(np.array(women.weight))


# #### （6）分析结果的可视化

# In[82]:


# 导入可视化包matplotlib.pyplot

import matplotlib.pyplot as plt

# 绘制原women数据的散点图
plt.scatter(women.height, women.weight)

# 绘制用训练模型根据women.height预测的predicted_weight
plt.plot(women.height, predicted_weight)

plt.rcParams['font.family']="SimHei" #显示汉字的方法
# 添加标题
plt.title('女性体重与身高的线性回归分析')

# 添加X轴名称
plt.xlabel('身高')

# 添加Y轴名称
plt.ylabel('体重')

# 显示绘图
plt.show()


# #### （7）生成报告

# In[84]:


# 重新绘制一遍图形，并将结果保存为PDF文件
# 若之前为调用show()方法，则可直接保存
# 可在调用show()方法之前绘制结果

# 绘制原women数据的散点图
plt.scatter(women.height, women.weight)

# 绘制用训练模型根据women.height预测的predicted_weight
plt.plot(women.height, predicted_weight)

# 添加标题
plt.title('女性体重与身高的线性回归分析')

# 添加X轴名称
plt.xlabel('身高')

# 添加Y轴名称
plt.ylabel('体重')

# 调用savefig()函数，保存会绘制结果
# 也可保存为其他格式，如png, jpg, svg等
plt.savefig('线性回归结果1.pdf')


# #### 6.3 机器学习
# 
# 【例1】KNN算法

# #### （1）数据读入

# In[85]:


bc_data = pd.read_csv('bc_data.csv', header=0)
    # 由于数据没有列名信息，header设置为None
bc_data.head()


# #### （2）数据理解

# In[86]:


# 查看描述性统计分析
print(bc_data.describe())


# In[87]:


# 查看列名
print(bc_data.columns)


# In[88]:


# 查看形状
print(bc_data.shape)


# #### （4）数据准备

# In[89]:


# 导入train_test_split()函数用于构建训练集和测试集
# 导入KNeighborsClassifier分类器
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 删除没有实际意义的ID项数据
data = bc_data.drop(['id'], axis=1)

# 查看删除后的数据项
print(data.head())


# In[92]:


# 获取特征矩阵
X_data = data.drop(['diagnosis'], axis=1)
X_data.head()


# In[94]:


# 获取结果数组
y_data = np.ravel(data[['diagnosis']])
    # np.ravel()用于降维处理
y_data[0:6]


# In[98]:


# 拆分测试数据与训练数据
# 用train_test_split()随机拆分训练集合测试集
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=1)

get_ipython().magic('pinfo train_test_split')


# In[99]:


# 查看训练数据与测试数据的数量
print(X_train.shape)
print(X_test.shape)


# #### （5）数据建模

# In[100]:


# 实例化KNN分类模型
model = KNeighborsClassifier(algorithm='kd_tree')


# In[101]:


# 用训练集训练模型
model.fit(X_train, y_train)


# In[103]:


# 用训练模型预测测试集数据
y_model = model.predict(X_test)


# In[104]:


# 查看预测结果和测试集的结果
print(y_model)


# In[105]:


print(y_test)


# In[106]:


# 计算预测准确率
np.mean(y_model == y_test)


# #### （6）模型准确率

# In[108]:


# 导入accaccuracy_score()函数用于计算模型的准确率
from sklearn.metrics import accuracy_score

# 查看模型的准确率
print(accuracy_score(y_test, y_model))


# 【例2】K-Means算法

# #### （1）数据导入

# In[109]:


# 读入数据
protein = pd.read_table('protein.txt', sep='\t')

# 查看前5条数据
protein.head()


# #### （2）数据理解

# In[110]:


# 查看描述性统计分析
print(protein.describe())


# In[111]:


# 查看列名
print(protein.columns)


# In[112]:


# 查看行数和列数
print(protein.shape)


# #### （3）数据转换
# 

# In[114]:


from sklearn import preprocessing

# 由于Country不是一个特征值，故舍去
sprotein = protein.drop(['Country'], axis=1)

# 对数据进行标准化处理
sprotein_scaled = preprocessing.scale(sprotein)

# 查看处理结果
print(sprotein_scaled)


# #### （4）数据建模

# In[117]:


# 导入KMeans类型
from sklearn.cluster import KMeans

# 实例化一个KMeans聚类器
kmeans = KMeans(n_clusters=5)
    # n_cluster为聚类中心
    


# In[130]:


# 训练模型
kmeans.fit(sprotein_scaled)


# #### （5）查看模型

# In[124]:


# 查看模型
print(kmeans)


# #### （6）模型预测

# In[125]:


# 预测聚类结果
y_kmeans = kmeans.predict(sprotein)
print(y_kmeans)


# #### （7）结果输出

# In[127]:


def print_kmcluster(k):
    '''用于聚类结果的输出
       k：为聚类中心个数
    '''
    for i in range(k):
        print('聚类', i)
        ls = []
        for index, value in enumerate(y_kmeans):
            if i == value:
                ls.append(index)
        print(protein.loc[ls, ['Country', 'RedMeat', 'Fish', 'Fr&Veg']])
            
print_kmcluster(5)          


# #### 6.4 数据可视化

# #### （1）数据准备

# In[128]:


# 读取数据
salaries = pd.read_csv('salaries.csv', index_col=0)


# In[61]:


# 查看数据
salaries.head()


# #### （2）导入Python包

# In[129]:


# 导入matplotlib.pyplot模块，并取别名为plt
import matplotlib.pyplot as plt
import seaborn as sns
# 设置行内显示图片
get_ipython().magic('matplotlib inline')


# #### （3）可视化绘图

# In[64]:


# 设置图片样式
sns.set_style('darkgrid')

# 绘制散点图
sns.stripplot(data=salaries, x='rank', y='salary', jitter=True, alpha=0.5)

# 绘制箱线图
sns.boxplot(data=salaries, x='rank', y='salary')

