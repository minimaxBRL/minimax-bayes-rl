import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

print(2.5 % 1)

# step_size = 0.5
# alpha_range = 1.5
# beta_range = 1.5
#
# alpha_list = [round(i * step_size, 1) for i in range(1, int(alpha_range / step_size) + 1)]
# beta_list = [round(i * step_size, 1) for i in range(1, int(beta_range / step_size) + 1)]
#
# ut = np.zeros([int(alpha_range / step_size), int(beta_range / step_size)])
#
# ut[0, 0] = 1
# ut[2, 1] = 1
# x = alpha_list
# y = beta_list
# x, y = np.meshgrid(x, y)
# print(x)
# print(y)
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# x = alpha_list
# y = beta_list
# x, y = np.meshgrid(x, y)
#
# # plot Bayes-expected regret
# ax.plot_surface(x, y, ut, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
#
# # don't know why, but apparently x-axis is betas and y-axis alphas
# ax.set_xlabel('betas')
# ax.set_ylabel('alphas')
# ax.set_zlabel('regret')
# ax.invert_xaxis()
# plt.show()