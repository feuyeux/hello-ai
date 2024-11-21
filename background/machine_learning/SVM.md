# SVM

## 原理

<https://person.zju.edu.cn/huhaoji>

<https://www.bilibili.com/video/BV1jt4y1E7BQ?vd_source=80dafe3ad5cf738ccf91d1abfbd8525a>

### 定义

1 训练数据及标签： $\left(X_1, y_1\right)\left(X_2, y_2\right)\cdots\left(X_N, y_N\right)$

$X$ 是特征向量 $\mathbf{X_1} = \begin{bmatrix}
x_{11} \\
x_{12} \\
\vdots \\
x_{1n}
\end{bmatrix}$

2 线性模型：$(\omega,b)$ $\mathbf{\omega}^\intercal\mathbf{X}+b=0$(超平面 Hyperplane)

$\omega$ 是权重向量 $\mathbf{\omega} = \begin{bmatrix}
\omega_1 \\
\omega_2 \\
\vdots \\
\omega_n
\end{bmatrix}$

$\mathbf{\omega}^T = \begin{bmatrix} \omega_1 & \omega_2 & \ldots & \omega_n \end{bmatrix}$

$b$是常数

### 一个训练集线性可分

对于集合存在模型，使任意元素，有标签为正例时，超平面大等于0，标签为负例时，超平面小于0

训练样本 $\{X_i, y_i\}\quad i=[1,N]$

$\exists(\omega,b)$，使：

对于$\forall i=1\textasciitilde N$，有：
$$
若 y_i=+1,则 \omega^\intercal X_i+b\ge 0 \\
若 y_i=-1,则 \omega^\intercal X_i+b\lt 0
$$
公式：$y_i[\omega^\intercal X_i+b] \ge 0$

### 支持向量机(Support Vector Machine)

SVM是最大化间隔(Margin)的分类算法

#### 优化问题（凸优化问题，二次规划问题）

最小化(Minimize)：$\|\mathbf{\omega}\|^2$

限制条件(Subject to)：$y_i[\omega^\intercal X_i+b] \ge 1 \, (i=1 \textasciitilde N)$

二次规划(Quadratic Programming)

目标函数(Objective Function) ：二次项

限制条件：一次项

要么无解，要么只有一个极值

### SVM处理非线性

1 最小化(Minimize)：$\frac{1}{2}\|\mathbf{\omega}\|^2+C \displaystyle \sum_{i=1}^{N} \xi_i(正则项(Regulatrion Term))$

限制条件(Subject to)：

$y_i[\omega^\intercal X_i+b] \ge 1- \xi_i \, (i=1 \textasciitilde N)$

$\xi_i \ge 0$

$C$ 实现设定的参数

$\xi_i$ 松弛变量(Slack Variable)

2 高维映射 $\phi(x)$

$x$ (低维) $-\phi\rightarrow $ $\phi(x)$ (高维)

#### 异或问题

$$
x_1=\begin{bmatrix}0\\0\\\end{bmatrix}\in C_1 \quad
x_2=\begin{bmatrix}1\\1\\\end{bmatrix}\in C_1 \\
x_3=\begin{bmatrix}0\\1\\\end{bmatrix}\in C_2 \quad
x_4=\begin{bmatrix}1\\0\\\end{bmatrix}\in C_2 \\
$$

##### 升维

可以是(不是唯一方式)：

$X=\begin{bmatrix}a\\b\\\end{bmatrix}$ (低维) $-\phi\rightarrow $ $\phi(X)=\begin{bmatrix}a^2\\b^2\\a\\b\\ab\end{bmatrix}$ (高维)

$$
\phi(x_1)=\begin{bmatrix}0\\0\\0\\0\\0\end{bmatrix} \quad
\phi(x_2)=\begin{bmatrix}1\\1\\1\\1\\1\end{bmatrix} \quad
\in C_1 \\
\phi(x_3)=\begin{bmatrix}1\\0\\1\\0\\0\end{bmatrix} \quad
\phi(x_4)=\begin{bmatrix}0\\1\\0\\1\\0\end{bmatrix} \quad
\in C_2
$$
$\omega$和$b$可以是(不是唯一解)：

$\phi(x_1)=\begin{bmatrix}-1\\-1\\-1\\-1\\6\end{bmatrix} \quad b=1$

#### 核函数

我们可以不知道无限维映射 $\phi(x)$的显式表达，我们只需要知道一个核函数(Kernel Function)

$$
K(X_1,X2)=\phi(X_1)^\intercal \phi(X_2)
$$

$\phi(X_1)$与$\phi(X_1)$两个无限维向量的内积，是一个数

##### 高斯核

$$
K(X_1,X_2)=e^{-\frac{\|X_1-X_2\|^2}{2\sigma^2}} \;
$$

##### 多项式核

$$
K(X_1,X_2)=({X_1}^\intercal{X_2}+1)^d \quad (d\,是多项式阶数)
$$

$K(X_1,X2)$ 能写成 $\phi(X_1)^\intercal \phi(X_2)$  的充要条件 (Mercer's Theorem)：

1 $K(X_1,X2)=K(X_2,X1)$ 【交换性】

2 $\forall C_i$ (常数)，$X_i$ (向量) (i=1~N)，有：【半正定性】

$\displaystyle \sum_{i=1}^N \displaystyle \sum_{j=1}^N C_iC_jK(X_i,X_j)\ge0$

### 优化理论

- Convex optimization
- Nonlinear Programming

#### 原问题(Prime Problem)  (非常普适)

最小化：$f(\omega)$

限制条件：

$g_i(\omega) \le 0 (i=[1,K])$

$h_i(\omega)=0 (i=[1,M])$

#### 对偶问题(Dual Problem)

##### 定义

> 代数转几何

$$
\begin{align*}
& L(\omega,\alpha,\beta) \\
& =f(\omega)+\displaystyle \sum_{i=1}^{K} \alpha_ig_i(\omega)+\displaystyle \sum_{i=1}^{M} \beta_ih_i(\omega) \\
& =f(\omega)+\alpha^\intercal g(\omega) + \beta^\intercal h(\omega)
\end{align*}
$$

$$
g(\omega)=\begin{bmatrix}g_1(\omega)\\g_2(\omega)\\\vdots\\g_K(\omega)\end{bmatrix} \quad
h(\omega)=\begin{bmatrix}h_1(\omega)\\h_2(\omega)\\\vdots\\h_M(\omega)\end{bmatrix}
$$

##### 对偶问题定义

最大化：$\theta(\alpha,\beta)=\inf_{所有\omega}\{L(\omega,\alpha,\beta)\}$

限制条件：$\alpha_i \ge 0 \, (i=[1,K])$

##### 定理

如果$\omega^*$是原问题的解，而$\alpha^*$,$\beta^*$是对偶问题的解，则有：
$$
\begin{align*}
& f(\omega^*) \ge \theta(\alpha^*,\beta^*) \\ \\
& \theta(\alpha^*,\beta^*)=\inf_{所有\omega}\{L(\omega,\alpha^*,\beta^*)\} \\
& \le L(\omega^*,\alpha^*,\beta^*) \\
& =f(\omega^*)+\displaystyle \sum_{i=1}^{K} \alpha^*_ig_i(\omega^*)+\displaystyle \sum_{i=1}^{M} \beta^*_ih_i(\omega^*) \\
& \le f(\omega^*)
\end{align*}
$$

$$
\begin{align*}
& \displaystyle \sum_{i=1}^{K} \alpha^*_i \ge 0 \\
& g_i(\omega^*) \le 0 \\
& \displaystyle \sum_{i=1}^{M} \beta^*_ih_i(\omega^*) = 0
\end{align*}
$$

定义：$G=f(\omega^*)-\theta(\alpha^*,\beta^*) \ge 0$

$G$叫做原问题与对偶问题的间距(Duality Gap)

对于某些特定优化问题，可以证明G=0

##### 强对偶定理

若$f(\omega)$是凸函数，且$g(\omega)=A\omega+b$，$h(\omega)=C\omega+d$，则此优化问题的原问题与对偶问题间距为0，即  $ f(\omega^*) = \theta(\alpha^*,\beta^*)$

对于$\forall i=[1,K]$ 或者 $\alpha^*_i=0$ 或者 $g^*_i(\omega^*)=0$ (KKT条件)
