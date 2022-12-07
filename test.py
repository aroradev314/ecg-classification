import funcs
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 3))
plt.plot(range(-5, 6), funcs.mex_func(2, -5, 6, 3, 0))
