import asyncio
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import urllib.request
from matplotlib import pyplot as plt
import sklearn
import numpy as np
from pathlib import Path
import pandas as pd
import sys
import platform
from common import image_path
from common import save_fig

np.random.seed(42)
print("python version:", sys.version_info)
print("sklearn version:", sklearn.__version__)

IMAGES_PATH = image_path()
if IMAGES_PATH is None:
    raise ValueError(
        "IMAGES_PATH is None. Please check the image_path function.")
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

# ==== Dataset ====

# the Better Life Index(BLI) data is in `datasets/lifesat/oecd_bli.csv` (data from 2020)
# the GDP per capita data is in `datasets/lifesat/gdp_per_capita.csv` (data up to 2020)
if platform.system() == "Windows":
    datapath = Path("d:/park/handson-ml3/datasets/lifesat")
elif platform.system() == "Linux" or platform.system() == "Darwin":
    datapath = Path.home() / "handson-ml3/datasets/lifesat"

data_root = "https://github.com/ageron/data/raw/main/"
datapath.mkdir(parents=True, exist_ok=True)
for filename in ("oecd_bli.csv", "gdp_per_capita.csv"):
    file_path = datapath / filename
    if not file_path.exists():
        print("Downloading", filename)
        url = data_root + "lifesat/" + filename
        urllib.request.urlretrieve(url, file_path)
    else:
        print(f"{filename} already exists. No download needed.")

oecd_bli = pd.read_csv(datapath / "oecd_bli.csv")
gdp_per_capita = pd.read_csv(datapath / "gdp_per_capita.csv")

gdp_year = 2020
gdppc_col = "GDP per capita (USD)"
lifesat_col = "Life satisfaction"

gdp_per_capita = gdp_per_capita[gdp_per_capita["Year"] == gdp_year]
gdp_per_capita = gdp_per_capita.drop(["Code", "Year"], axis=1)
gdp_per_capita.columns = ["Country", gdppc_col]
gdp_per_capita.set_index("Country", inplace=True)
gdp_per_capita.head()

oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
oecd_bli.head()

# merge the life satisfaction data and the GDP per capita data, keeping only the GDP per capita and Life satisfaction columns:
full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                              left_index=True, right_index=True)
full_country_stats.sort_values(by=gdppc_col, inplace=True)
full_country_stats = full_country_stats[[gdppc_col, lifesat_col]]
full_country_stats.head()

# To illustrate the risk of overfitting, I use only part of the data in most figures(all countries with a GDP per capita between `min_gdp` and `max_gdp`). Later in the chapter I reveal the missing countries, and show that they don't follow the same linear trend at all.
min_gdp = 23_500
max_gdp = 62_500
country_stats = full_country_stats[(full_country_stats[gdppc_col] >= min_gdp) &
                                   (full_country_stats[gdppc_col] <= max_gdp)]
country_stats.head()

full_country_stats.to_csv(datapath / "lifesat_full.csv")
country_stats.to_csv(datapath / "lifesat.csv")

country_stats.plot(kind='scatter', figsize=(5, 3), grid=True,
                   x=gdppc_col, y=lifesat_col)

min_life_sat = 4
max_life_sat = 9

position_text = {
    "Turkey": (29_500, 4.2),
    "Hungary": (28_000, 6.9),
    "France": (40_000, 5),
    "New Zealand": (28_000, 8.2),
    "Australia": (50_000, 5.5),
    "United States": (59_000, 5.3),
    "Denmark": (46_000, 8.5)
}

for country, pos_text in position_text.items():
    pos_data_x = country_stats[gdppc_col].loc[country]
    pos_data_y = country_stats[lifesat_col].loc[country]
    country = "U.S." if country == "United States" else country
    plt.annotate(country, xy=(pos_data_x, pos_data_y),
                 xytext=pos_text, fontsize=12,
                 arrowprops=dict(facecolor='black', width=0.5,
                                 shrink=0.08, headwidth=5))
    plt.plot(pos_data_x, pos_data_y, "ro")

plt.axis([min_gdp, max_gdp, min_life_sat, max_life_sat])

save_fig('money_happy_scatterplot', IMAGES_PATH)
plt.show()

highlighted_countries = country_stats.loc[list(position_text.keys())]
highlighted_countries[[gdppc_col, lifesat_col]].sort_values(by=gdppc_col)

country_stats.plot(kind='scatter', figsize=(5, 3), grid=True,
                   x=gdppc_col, y=lifesat_col)

# Return evenly spaced numbers over a specified interval
X = np.linspace(min_gdp, max_gdp, 1000)

w1, w2 = 4.2, 0
plt.plot(X, w1 + w2 * 1e-5 * X, "r")
plt.text(40_000, 4.9, fr"$\theta_0 = {w1}$", color="r")
plt.text(40_000, 4.4, fr"$\theta_1 = {w2}$", color="r")

w1, w2 = 10, -9
plt.plot(X, w1 + w2 * 1e-5 * X, "g")
plt.text(26_000, 8.5, fr"$\theta_0 = {w1}$", color="g")
plt.text(26_000, 8.0, fr"$\theta_1 = {w2} \times 10^{{-5}}$", color="g")

w1, w2 = 3, 8
plt.plot(X, w1 + w2 * 1e-5 * X, "b")
plt.text(48_000, 8.5, fr"$\theta_0 = {w1}$", color="b")
plt.text(48_000, 8.0, fr"$\theta_1 = {w2} \times 10^{{-5}}$", color="b")

plt.axis([min_gdp, max_gdp, min_life_sat, max_life_sat])

save_fig('tweaking_model_params_plot', IMAGES_PATH)
plt.show()

# ==== Model ====
# life_satisfaction = θ_0 + θ_1 × GDP_per_capita

# LinearRegression Model
X_sample = country_stats[[gdppc_col]].values
y_sample = country_stats[[lifesat_col]].values

# 最小二乘线性回归
# 使用系数w =（w1，…，wp）拟合线性模型，以最小化数据集中实际目标(observed targets)值与通过线性逼近(linear approximation)预测的目标(targets predicted)之间的残差平方和(residual sum)
lin1 = LinearRegression()
# 拟合|训练
lin1.fit(X_sample, y_sample)

# y = β0 + β1×1 + β2×2 + … + βnxn + ε
# y 是因变量（预测目标）
# x1, x2, …, xn 是自变量（特征）
# β0, β1, …, βn 是回归系数（模型的参数）
# ε 是误差项，代表不能被自变量解释的随机误差
# 截距：是线性回归方程中回归线与y轴的交点位置
# 线性回归方程的目标是找到一组最佳的回归系数 β0, β1, …, βn，使得通过线性组合自变量得到的模型预测值能够最好地拟合实际观测值。
# 拟合效果的好坏可以通过相关系数和截距来评估。

# theta_0 线性模型中的截距项(Independent term)。如果设置fit_intercept = False，则截距为0.0
intercept = lin1.intercept_
# theta_1 线性回归问题的估计系数(Estimated coefficients)。
# 如果在拟合过程中传递了多个目标（y 2D），则这是一个二维数组，形状为(n_targets, n_features)，而如果仅传递了一个目标，则是长度为n_features的一维数组。
coef = lin1.coef_

t0, t1 = intercept[0], coef[0][0]
print(f"θ0={t0:.2f}, θ1={t1:.2e}")

country_stats.plot(
    kind='scatter',
    # a tuple (width, height) in inches
    figsize=(5, 3),
    grid=True,
    x=gdppc_col,
    y=lifesat_col
)

X = np.linspace(min_gdp, max_gdp, 1000)
plt.plot(X, t0 + t1 * X, "b")
plt.text(max_gdp - 20_000, min_life_sat + 1.9,
         fr"$\theta_0 = {t0:.2f}$", color="b")
plt.text(max_gdp - 20_000, min_life_sat + 1.3,
         fr"$\theta_1 = {t1 * 1e5:.2f} \times 10^{{-5}}$", color="b")
plt.axis([min_gdp, max_gdp, min_life_sat, max_life_sat])
save_fig('best_fit_model_plot', IMAGES_PATH)
plt.show()

# 预测
cyprus_gdp_per_capita = gdp_per_capita[gdppc_col].loc["Cyprus"]
cyprus_predicted_life_satisfaction = lin1.predict(
    [[cyprus_gdp_per_capita]])[0, 0]
cyprus_predicted_life_satisfaction
print(
    f"cyprus_gdp_per_capita={cyprus_gdp_per_capita}, cyprus_predicted_life_satisfaction={cyprus_predicted_life_satisfaction}")

country_stats.plot(
    kind='scatter',
    figsize=(5, 3),
    grid=True,
    x=gdppc_col,
    y=lifesat_col
)

X = np.linspace(min_gdp, max_gdp, 1000)
plt.plot(X, t0 + t1 * X, "b")
plt.text(min_gdp + 22_000, max_life_sat - 1.1,
         fr"$\theta_0 = {t0:.2f}$", color="b")
plt.text(min_gdp + 22_000, max_life_sat - 0.6,
         fr"$\theta_1 = {t1 * 1e5:.2f} \times 10^{{-5}}$", color="b")
plt.plot([cyprus_gdp_per_capita, cyprus_gdp_per_capita],
         [min_life_sat, cyprus_predicted_life_satisfaction], "r--")
plt.text(cyprus_gdp_per_capita + 1000, 5.0,
         fr"Prediction = {cyprus_predicted_life_satisfaction:.2f}", color="r")
plt.plot(cyprus_gdp_per_capita, cyprus_predicted_life_satisfaction, "ro")
plt.axis([min_gdp, max_gdp, min_life_sat, max_life_sat])
plt.show()

missing_data = full_country_stats[(full_country_stats[gdppc_col] < min_gdp) |
                                  (full_country_stats[gdppc_col] > max_gdp)]
position_text_missing_countries = {
    "South Africa": (20_000, 4.2),
    "Colombia": (6_000, 8.2),
    "Brazil": (18_000, 7.8),
    "Mexico": (24_000, 7.4),
    "Chile": (30_000, 7.0),
    "Norway": (51_000, 6.2),
    "Switzerland": (62_000, 5.7),
    "Ireland": (81_000, 5.2),
    "Luxembourg": (92_000, 4.7),
}

full_country_stats.plot(kind='scatter', figsize=(8, 3),
                        x=gdppc_col, y=lifesat_col, grid=True)

for country, pos_text in position_text_missing_countries.items():
    pos_data_x, pos_data_y = missing_data.loc[country]
    plt.annotate(country, xy=(pos_data_x, pos_data_y),
                 xytext=pos_text, fontsize=12,
                 arrowprops=dict(facecolor='black', width=0.5,
                                 shrink=0.08, headwidth=5))
    plt.plot(pos_data_x, pos_data_y, "rs")

X = np.linspace(0, 115_000, 1000)
plt.plot(X, t0 + t1 * X, "b:")

lin_reg_full = LinearRegression()
Xfull = np.c_[full_country_stats[gdppc_col]]
yfull = np.c_[full_country_stats[lifesat_col]]
lin_reg_full.fit(Xfull, yfull)

t0full, t1full = lin_reg_full.intercept_[0], lin_reg_full.coef_[0][0]
X = np.linspace(0, 115_000, 1000)
plt.plot(X, t0full + t1full * X, "k")
plt.axis([0, 115_000, min_life_sat, max_life_sat])
save_fig('representative_training_data_scatterplot', IMAGES_PATH)
plt.show()

full_country_stats.plot(kind='scatter', figsize=(8, 3),
                        x=gdppc_col, y=lifesat_col, grid=True)
poly = preprocessing.PolynomialFeatures(degree=10, include_bias=False)
scaler = preprocessing.StandardScaler()
lin_reg2 = LinearRegression()
pipeline_reg = pipeline.Pipeline([
    ('poly', poly),
    ('scal', scaler),
    ('lin', lin_reg2)])
pipeline_reg.fit(Xfull, yfull)
curve = pipeline_reg.predict(X[:, np.newaxis])
plt.plot(X, curve)
plt.axis([0, 115_000, min_life_sat, max_life_sat])
save_fig('overfitting_model_plot', IMAGES_PATH)
plt.show()

w_countries = [c for c in full_country_stats.index if "W" in c.upper()]
full_country_stats.loc[w_countries][lifesat_col]

all_w_countries = [c for c in gdp_per_capita.index if "W" in c.upper()]
gdp_per_capita.loc[all_w_countries].sort_values(by=gdppc_col)


country_stats.plot(kind='scatter', x=gdppc_col, y=lifesat_col, figsize=(8, 3))
missing_data.plot(kind='scatter', x=gdppc_col, y=lifesat_col,
                  marker="s", color="r", grid=True, ax=plt.gca())

X = np.linspace(0, 115_000, 1000)
plt.plot(X, t0 + t1*X, "b:", label="Linear model on partial data")
plt.plot(X, t0full + t1full * X, "k-", label="Linear model on all data")

# 岭回归 在线性回归的基础上增加了一个正则化项，以减小特征的系数，从而降低模型的复杂度
# Linear least squares with l2 regularization
ridge = Ridge(alpha=10**9.5)
X_sample = country_stats[[gdppc_col]]
y_sample = country_stats[[lifesat_col]]
ridge.fit(X_sample, y_sample)
t0ridge, t1ridge = ridge.intercept_[0], ridge.coef_[0][0]
plt.plot(X, t0ridge + t1ridge * X, "b--",
         label="Regularized linear model on partial data")
plt.legend(loc="lower right")
plt.axis([0, 115_000, min_life_sat, max_life_sat])
save_fig('ridge_model_plot', IMAGES_PATH)
plt.show()


async def main():
    await asyncio.sleep(2)
    print("Done.")
asyncio.run(main())