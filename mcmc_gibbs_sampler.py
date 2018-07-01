import numpy as np
import numpy.random as rd
from scipy.stats import invgamma
from scipy.stats import norm
import matplotlib.pyplot as plt

n = 1000
data = rd.normal(10, 2, size=n)

mean = 0
std = 10
params = [(mean, std)]

# 平均の事前分布は正規分布で、
# 分散の事前分布は逆ガンマ分布となるとする

# ハイパーパラメータの設定
mu0 = 0
m0 = 1
alpha0 = 0.02
beta0 = 0.02

burnin = 4000
cnt = 11000

for t in params:
    if cnt == 0:
        break
    
    # ハイパーパラメータの計算
    m1 = m0 + n
    mu1 = (m0 * mu0 + n * np.mean(data)) / (m0 + n)
    alpha1 = alpha0 + n
    # ここで正規分布から平均を生成
    mean = rd.normal(mu1, t[1] / m1)
    # これもハイパーパラメータの更新
    beta1 = beta0 + n * np.std(data) + n * m0 * (mean - mu0) ** 2 / (n + m0)
    # ここで逆ガンマ分布から分散を生成
    std = invgamma.rvs((alpha1 + 1) / 2.0)
    # 生成した値をスケーリング
    std = std * (m1 * (mean - mu1) ** 2 + beta1) / 2.0
    # タプルを記録
    params.append((mean, std))
    # ハイパーパラメータの更新
    mu0 = mu1
    m0 = m1
    alpha0 = alpha1
    beta0 = beta1
    # カウントダウン
    cnt = cnt - 1
    
# 結果の表示
x = np.arange(11001)
means = []
stds = []
for i in params:
    means.append(i[0])
    stds.append(i[1])

x = x[burnin + 1:]
means = means[burnin + 1:]
stds = stds[burnin + 1:]
print(np.mean(means))
print(np.mean(stds))

plt.subplot(2, 1, 1)
plt.plot(x, means)
plt.subplot(2, 1, 2)
plt.plot(x, stds)

print("-------------------------")
print("|----\----| mean |st.dev.|")
print("| expected|{:.3f}|{:.3f}|".format(np.mean(data), np.std(data)))
print("|predicted|{:.3f}|{:.3f}|".format(np.mean(means),np.mean(stds)))
print("-------------------------")

print("-------------------------")
print("|parameter| mean |st.dev.|    95% CI    |")
print("|    mu   |{:.2f}|{:.2f}|({:.2f},{:.2f})|".format(
        np.mean(means),
        np.std(means),
        norm.ppf(q=0.025, loc=np.mean(means), scale=np.sqrt(np.std(means))),
        norm.ppf(q=0.975, loc=np.mean(means), scale=np.sqrt(np.std(means)))))
print("|   std   |{:.2f}|{:.2f}|({:.2f},{:.2f})|".format(
        np.mean(stds),
        np.std(stds),
        norm.ppf(q=0.025, loc=np.mean(stds), scale=np.sqrt(np.std(stds))),
        norm.ppf(q=0.975, loc=np.mean(stds), scale=np.sqrt(np.std(stds)))))