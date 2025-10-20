#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

#%%
res1 = pd.read_csv("../../data/ElectricityPrice2030.csv").rename(columns={"price":"Price 2030"})
res2 = pd.read_csv("../../data/ElectricityPrice2050.csv").rename(columns={"price":"Price 2050"})
res3 = pd.read_csv("../../data/ElectricityPrice2023.csv").rename(columns={"Day-ahead Price (EUR/MWh)":"Price 2023"})

res3 = res3[res3["Sequence"] == "Sequence Sequence 1"].set_index("Time")["Price 2023"]
res3.index = pd.to_datetime(res3.index,dayfirst=True)
res3 = res3.resample("h").mean().reset_index()
print(res3)

#%%
res4 = pd.read_csv("../../data/LoadProfile.csv").reset_index()["Load Profile"]
# print(res3["load"])
res5 = pd.concat([res3,res1,res2,res4/7],axis=1).set_index("Time")
fig = (res5/res5.std(axis=0)).plot(linewidth = 0.5, alpha = 0.8)
fig.grid()
fig.set_title("Input variations")

#%%
fig, ax = plt.subplots()
colors = ['red','green','blue']

x_axis = np.arange(res5['Load Profile'].min(), res5['Load Profile'].max())
for col, color in zip([col for col in res5.columns if col != "Load Profile"],colors):
    fig = (res5).plot(  x= 'Load Profile', 
                        y=col,
                        kind='scatter',
                        linewidth=0.5, 
                        alpha=0.1, 
                        ax = ax, 
                        label = col,
                        color = color)
    res = stats.linregress(res5['Load Profile'], y = res5[col])
    R2 = res.rvalue**2
    print(f"R-squared: {R2:.6f}")
    f = lambda x: res.intercept + res.slope*x
    plt.plot(x_axis, f(x_axis) color, label=R2)
    # plt.plot(f(x_axis), interpolation, color = )
ax.set_ylim(0,300)
ax.legend()
ax.grid()

# %%
res6 = res5.copy().drop(columns = "Load Profile")
print(res6)
res6["Month"] = res6.index.month
# import matplotlib.pyplot as plt

numeric_cols = res6.select_dtypes(include="number").columns

# melt wide data into long form
res6_long = res6.melt(id_vars="Month", var_name="Year", value_name="Price EUR/MWh")

plt.figure(figsize=(12,6))
sns.boxplot(data=res6_long, x="Month", y="Price EUR/MWh", hue="Year")
plt.title("Monthly Distribution of Electricity prices (EUR/MWh)")
plt.grid()
plt.xticks([-0.5+i for i in range(12)])  # Grid positions
plt.ylim(-10,300)
plt.savefig("Monthly_Price_Evolution_Austria.pdf")
plt.show()

# Maybe try to capture the transition for the typical weeks from 2023 to 2050, and ensure that the variance is just as continuous of average.


# %%
