#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import linregress

#%%
price_type = "high"
res1 = pd.read_csv(f"../../data/price_{price_type}.csv")#.rename(columns={"price":"Price 2030"})
# res2 = pd.read_csv("../../data/ElectricityPrice2050.csv").rename(columns={"price":"Price 2050"})
# res3 = pd.read_csv("../../data/ElectricityPrice2023.csv").rename(columns={"Day-ahead Price (EUR/MWh)":"Price 2023"})

# res3 = res3[res3["Sequence"] == "Sequence Sequence 1"].set_index("Time")["Price 2023"]
# res3.index = pd.to_datetime(res3.index,dayfirst=True)
# res3 = res3.resample("h").mean().reset_index()
# print(res1)
res2 = res1[["2025","2030","2035","2040","2045","2050"]]

# Create datetime index for a specific year
start_date = '2023-01-01'
end_date = '2023-12-31 23:00'
datetime_index = pd.date_range(start=start_date, end=end_date, freq='H')

# Set the index for your dataframe
res2["Time"] = datetime_index

#%%
res4 = pd.read_csv("../../data/LoadProfile.csv").reset_index()["Load Profile"]
# print(res3["load"])
res5 = pd.concat([res2,res4/7],axis=1).set_index("Time")
fig = (res5/res5.std(axis=0)).plot(linewidth = 0.5, alpha = 0.8)
fig.grid()
fig.set_title("Input variations")

#%%
fig, ax = plt.subplots()

colors = plt.cm.Reds(np.linspace(0.1, 1, len(res5.columns)))

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
    res = linregress(res5['Load Profile'], y = res5[col])
    R2 = res.rvalue**2
    print(f"R-squared: {R2:.6f}")
    f = lambda x: res.intercept + res.slope*x
    print(color)
    ax.plot(x_axis, f(x_axis), color, label=f"R-squared: {R2:.2f}")
    # plt.plot(f(x_axis), interpolation, color = )
# ax.set_ylim(0,300)
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
plt.title(f"Monthly Distribution of Electricity {price_type} prices (EUR/MWh)")
plt.grid()
plt.xticks([-0.5+i for i in range(12)])  # Grid positions
# plt.ylim(-10,300)
plt.savefig(f"Monthly_Price_Evolution_Austria_{price_type}.pdf")
plt.show()

# Maybe try to capture the transition for the typical weeks from 2023 to 2050, and ensure that the variance is just as continuous of average.


# %%
res7 = res5.copy()
print(res7)
# Aggregate weekly and study mean
# res7.drop(columns = ["Load Profile"], inplace = True)
res8_mean = res7.groupby(pd.Grouper(freq='W')).mean() 
res8_std = res7.groupby(pd.Grouper(freq='W')).std() 

ax = res8_mean.plot(figsize=(12, 6), linewidth=2)

# Get the colors used by pandas plotting
lines = ax.get_lines()
colors = [line.get_color() for line in lines]

# Add shaded areas for each column
for i, column in enumerate(res8_mean.columns):
    ax.fill_between(res8_mean.index,
                   res8_mean[column] - res8_std[column],
                   res8_mean[column] + res8_std[column],
                   alpha=0.3, color=colors[i])

ax.legend()
ax.set_title(f'{price_type} Price Weekly Mean with Standard Deviation for Representative Year')
plt.savefig(f'price_mean_var-{price_type}.pdf')
plt.show()
# res8_mean.plot()


# Aggregate weekly and study variance
# Aggregate weekly and study self correlation: average vertically and extract self-correlation mean and variance

#%%

# print((res7).isna)

# Check for missing values
print("\nMissing values per column:")
print(res7["Price 2023"].isnull().sum())

# Check data types
print("\nData types:")
print(res7["Price 2023"].dtypes)

# Check for infinite values
print("\nInfinite values:")
print(np.isinf(res7["Price 2023"]).sum())

# Check for constant series (zero variance)
print("\nStandard deviation per column:")
print(res7["Price 2023"].std())

acf(res7["Price 2023"])
#%%
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import matplotlib.collections as mc

horizon = 4*168
fig, axs = plt.subplots(len(res7.columns), 1, sharex=True, figsize=(2*len(res7.columns), 6))
for i, column in enumerate(res7.columns):
    # Draw ACF plot
    plot_acf(res7[column].dropna(), lags=horizon, ax=axs[i], title=f'ACF for {column} CF for {column}')
    
    # --- Remove only the vertical bars (LineCollection) ---
    for collection in list(axs[i].collections):
        if isinstance(collection, mc.LineCollection):
            collection.remove()
    
    # --- Keep line/marker for the ACF values ---
    for line in axs[i].lines:
        line.set_linestyle('-')     # use 'None' for dots only
        line.set_marker('o')
        line.set_markersize(4)
        line.set_color('C0')
    
    axs[i].grid(True, linestyle='--', alpha=0.6)
    # axs[i].set_tight_layout()
    axs[i].set_xlim(0,horizon)
    axs[i].set_ylim(0,1)
    axs[i].set_ylabel("Autocorrelation")
    # axs[i].show()


axs[i].set_xlabel("Lags (hours)")
fig.savefig(f"Autocorrelation_Analysis_global-{price_type}.pdf")

# %%
res7 = res5.copy()
print(res7)
# res7.drop(columns = ["Load Profile"], inplace = True)
# Alternative approach using pandas autocorr method
def calculate_weekly_autocorr_pandas(week_data):
    """Calculate autocorrelation using pandas method"""
    if len(week_data) < 168:
        return np.full(168, np.nan)
    
    autocorrs = []
    for lag in range(168):
        if lag < len(week_data):
            autocorrs.append(week_data.autocorr(lag=lag))
        else:
            autocorrs.append(np.nan)
    
    return autocorrs

# Calculate using the alternative method
weekly_autocorrs_alt = {}

for column in res7.columns:
    weekly_data = []
    weekly_groups = res7[column].groupby(pd.Grouper(freq='W'))
    
    for week_label, week_series in weekly_groups:
        if len(week_series) >= 168:
            autocorr = calculate_weekly_autocorr_pandas(week_series)
            weekly_data.append(autocorr)
    
    weekly_autocorrs_alt[column] = weekly_data

# Aggregate and plot (same as before)
autocorr_stats_alt = {}

for column, autocorr_list in weekly_autocorrs_alt.items():
    if len(autocorr_list) > 0:
        autocorr_array = np.array(autocorr_list)
        mean_autocorr = np.nanmean(autocorr_array, axis=0)
        std_autocorr = np.nanstd(autocorr_array, axis=0)
        
        autocorr_stats_alt[column] = {
            'mean': mean_autocorr,
            'std': std_autocorr,
            'n_weeks': len(autocorr_list)
        }

# Plot alternative results
plt.figure(figsize=(14, 8))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, (column, stats) in enumerate(autocorr_stats_alt.items()):
    color = colors[i % len(colors)]
    hours = np.arange(168)
    
    plt.plot(hours, stats['mean'], 
             label=f'{column} ({stats["n_weeks"]} weeks)', 
             color=color, linewidth=2)
    
    plt.fill_between(hours, 
                    stats['mean'] - stats['std'], 
                    stats['mean'] + stats['std'], 
                    alpha=0.3, color=color)

plt.xlabel('Lag (hours)')
plt.ylabel('Autocorrelation')
plt.title('Weekly Autocorrelation (Pandas Method) with Standard Deviation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(1, 168)
plt.ylim(-1, 1)
plt.hlines(y=0,xmin = 0,xmax = 168, color = 'k')
plt.savefig(f"Autocorrelation-{price_type}.pdf")
plt.show()
# %%
