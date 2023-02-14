import numpy as np
import matplotlib.pyplot as plt

p = 0.5
x = 0.01
n = 1


K = 100

f_range = np.arange(K)

p = (1 / K) * (1 / (1 - (1 - x**n) * (1 - p)**f_range))
p_lb = (1 / K) * (1 / ((1 + f_range * (1 - x**n) * np.log(1 / (1 - p)))))


plt.plot(f_range, np.log(p), label='Pr[F = f]')
plt.plot(f_range, np.log(p_lb), label='LB')
plt.legend()
plt.show()
