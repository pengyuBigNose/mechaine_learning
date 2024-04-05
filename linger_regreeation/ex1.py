import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 最小二乘法拟合直线
data1 = np.array([[32,31],[53,68],[61,62],[47,71],[59,87],[55,78],[52,79],[39,59],[48,75],[52,71],
          [45,55],[54,82],[44,62],[58,75],[56,81],[48,60],[44,82],[60,97],[45, 48],[38,56],
          [66,83],[65,118],[47,57],[41,51],[51,75],[59,74],[57,95],[63,95],[46,79],[50,83]])
# 获取打印出data的属性
# print("data的属性：",data.shape)
# 如何随机生成n行2列的数据
# data1 = np.random.rand(100,2)
# print("随机生成的数据：",data1.shape)
# 如何生成数据集？
# 1. 随机生成数据
# 2. 从已有数据集中抽取数据
# 3. 手工输入数据

# 分离x,y
x = data1[:,0]
y = data1[:,1]

# 定义损失函数
# 损失函数是系数的函数，即L(w,b) = 1/2*sum((w*x+b-y)^2)
def Loss(w,b,points):
  total_loss = 0
  M = len(points)
  #逐点计算损失误差
  for i in range(M):
    x_i = points[i][0]
    y_i = points[i][1]
    total_loss += (w*x_i+b-y_i)**2

  return total_loss/M
# 定义算法拟合函数
#先定义一个求平均值的函数
def average(data):
  lens = len(data)
  sum = 0
  for i in range(lens):
    sum += data[i]
  return sum/lens

# 定义最小二乘法拟合函数
# 输入：数据集points
# 输出：拟合参数w,b
def fitM(points):
  M=len(points)
  x_bar=average(points[:,0])
  sum_xy=0
  sum_x2=0
  sum_delta=0
  for i in range(M):
    x=points[i,0]
    y=points[i,1]
    sum_xy+=(x-x_bar)*y
    sum_x2+=x**2
  #计算w
  w=sum_xy/(sum_x2-M*(x_bar**2))
  for i in range(M):
    x=points[i,0]
    y=points[i,1]
    sum_delta+=(y-w*x)
    
  #计算b
  b=sum_delta/M
  return w,b

# 定义梯度下降算法
# 输入：数据集points，学习率learning_rate，最大迭代次数max_iter
# 输出：拟合参数w,b，损失函数列表loss_list
def fit(points,learning_rate=0.01,max_iter=1000):
  # 初始化参数
  w = 0
  b = 0
  # 迭代次数
  iter_num = 0
  # 损失函数列表
  loss_list = []
  # 开始迭代
  while iter_num < max_iter:
    # 计算梯度
    grad_w = 0
    grad_b = 0
    M = len(points)
    for i in range(M):
      x_i = points[i][0]
      y_i = points[i][1]
      grad_w += (w*x_i+b-y_i)*x_i
      grad_b += (w*x_i+b-y_i)
    # 更新参数
    w -= learning_rate*grad_w/M
    b -= learning_rate*grad_b/M
    # 计算损失函数
    loss = Loss(w,b,points)
    # 记录损失函数
    loss_list.append(loss)
    # 打印信息
    #print("iter_num:",iter_num,"loss:",loss,"w:",w,"b:",b)
    # 迭代次数加1
    iter_num += 1
  # 返回参数
  return w,b,loss_list


# print("原始数据：",data)
# 调用拟合函数
#w,b,loss_list = fit(data)
w,b=fitM(data1)
# 打印参数
print("拟合参数：",w,b)
#拟合曲线
cost = Loss(w,b,data1)
print("拟合的损失函数：",cost)

# 画图
plt.scatter(x, y)
pred_y=w*x+b
plt.plot(x,pred_y,color='r')
plt.title("Linger Regression")
plt.xlabel("x")
plt.ylabel("y")

plt.show()




