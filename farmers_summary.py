import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# with open('results_market.pickle', 'rb') as f:
#     results_market = pickle.load(f)
#
# with open('results_profits.pickle', 'rb') as f:
#     results_profits = pickle.load(f)

with open('results_market_gov.pickle', 'rb') as f:
    results_market = pickle.load(f)

with open('results_profits_gov.pickle', 'rb') as f:
    results_profits = pickle.load(f)

rounds = results_market['Round'].max()

profits_calc = results_profits.loc[:, ~results_profits.columns.isin(['Round', 'Iteration'])]
market_calc = results_market.loc[:, ~results_market.columns.isin(['Round', 'Iteration'])]

results_profits['mean'] = profits_calc.mean(axis = 1)

for m in list(market_calc.stack().unique()):
    results_profits['mean_' + m] = profits_calc[market_calc.eq(m)].mean(axis = 1)
    results_profits['count_' + m] = market_calc[market_calc.eq(m)].count(axis = 1)

differences = 100 - (market_calc[1:].reset_index(drop=True) == market_calc[:-1].reset_index(drop=True)).sum(axis = 1)
differences.loc[results_profits['Iteration'] == 1000] = np.nan
results_profits['changes'] = differences

print(results_profits)

summary_columns = ['Iteration', 'mean', 'mean_W', 'mean_C', 'mean_S', 'mean_R', 'mean_G', 'count_W', 'count_S', 'count_R', 'count_C', 'count_G', 'changes']
summary_data = results_profits[summary_columns].groupby(['Iteration']).agg({'mean'})
summary_data.columns = list(map('_'.join, summary_data.columns.values))
summary_data.columns = summary_columns[1:]

print(summary_data)

agent_avg_profits = []
agent_avg_changes = []

for i in range(rounds + 1):
    market = results_market.loc[results_market['Round'] == i, results_market.columns.isin(list(range(100)))].copy()
    profits = results_profits.loc[results_profits['Round'] == i, results_profits.columns.isin(list(range(100)))].copy()

    agent_avg_profits.append(list(profits.mean(axis = 0)))
    agent_avg_changes.append(list(1 - ((market[1:].reset_index(drop=True) == market[:-1].reset_index(drop=True)).sum(axis = 0) / len(market))))

# plot1 - Profits vs Iteration
tmp = summary_data[['mean', 'mean_W', 'mean_C', 'mean_S', 'mean_R', 'mean_G']]
tmp.columns = ['profit_avg', 'profit_avg_wheat', 'profit_avg_corn', 'profit_avg_soybeans', 'profit_avg_rice', 'profit_avg_subsidy']

ax = tmp.plot(
    title = 'Average profit by round (n_Farmers = 100, n_Iterations = 100)',
    colormap = 'Set1',
    grid = True
)
ax.set_ylabel('Profit')
ax.set_xlabel('Round')
plt.savefig('gov_Profit_Iter.png', dpi = 200)
plt.show()

#plot2 - Number of farmers in each market at the and of a simulation
tmp  = pd.DataFrame({
'Market' : ['Wheat', 'Soybeans', 'Rice', 'Corn', 'Subsidy'],
'Farmers' : summary_data.loc[summary_data.index == 1000, ['count_W', 'count_S', 'count_R', 'count_C', 'count_G']].values.flatten().tolist()
})

ax = tmp.plot.bar(
    title = 'Average number of farmers in markets in 1000th round',
    legend = False,
    x = 'Market',
    y = 'Farmers',
    rot = 0,
    colormap = 'Set1'
)
ax.set_xlabel('Market')
ax.set_ylabel('Farmers')
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))


plt.savefig('gov_Market_Farmers_bar.png', dpi = 200)
plt.show()

#plot3 - Number of decision changes
ax = summary_data[['changes']].plot(
    title = 'Average decision changes by round (n_Farmers = 100, n_Iterations = 100)',
    colormap = 'Set1',
    grid = True
)
ax.set_ylabel("% of farmers who changed their decision")
ax.set_xlabel('Round')
plt.savefig('gov_Decision_chg.png', dpi = 200)
plt.show()

#plot4 - % of decision changes for each farmer vs its profits
def flatten(xss):
    return [x for xs in xss for x in xs]

import seaborn as sns
ax = sns.regplot(x=[x * 100 for x in flatten(agent_avg_changes)]
, y=flatten(agent_avg_profits),
    scatter_kws={'s':2},
    line_kws=dict(color = 'r')

)
ax.set(title = 'Profits vs. % of changed decisions (by Farmer)')
ax.set_ylabel('Profits')
ax.set_xlabel('% of changed decisions')
plt.savefig('gov_Profits_chg.png', dpi = 200)
plt.show()
