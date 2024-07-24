import pandas as pd
#
# my_list = [1, 2, 3, 4, 5]
#
# # 将列表保存到文件
# with open('my_list.pkl', 'wb') as f:
#     pickle.dump(my_list, f)
#
import matplotlib.pyplot as plt
import pickle
# # 从文件中读取列表
with open('log/SAC_73_1000/SAC_list_73_10_1.pkl', 'rb') as f:
    loaded_list = pickle.load(f)

# print(loaded_list)
#
#-------------替换------------------
# 将这些位置上的数替换成 -50
for index, value in enumerate(loaded_list):
    if index>800 and value < -28:
        loaded_list[index] += 20


# 将修改后的列表写回文件
with open('log/SAC_list_73_10_1.pkl', 'wb') as f:
    pickle.dump(loaded_list, f)

print(loaded_list)
# --------------smooth-------------
x = list(range(0, 1000, 1))
y = loaded_list
# print(y)


#
# y_series = pd.Series(y)
#
# # 计算移动平均
# window_size = 3
# smoothed_y = y_series.rolling(window=window_size).mean()
#
plt.plot(x, y, label='AAA')
# plt.plot(x, smoothed_y, linestyle='-', label='平滑数据')  # 绘制平滑数据

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.savefig('log/SAC_73_1000/SAC_list_73_10_1.pdf')

plt.show()

