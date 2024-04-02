import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

colors_hex = [ '#24325F', '#82491E', '#B7E4F9', '#E89242','#FB6467', '#69C8EC'] #, '#E762D7' , '#FAE48B',  '#A6EEE6', '#917C5D', '#526E2D', '#FAFD7C']
colors = [tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) for h in colors_hex]
cm = ListedColormap(colors_hex)


# 1000 rounds - in single experiment run
# 100 iterations - number of experiment runs

mg_dict = {
 'W' : 'g1',
 'C' : 'g2',
 'S' : 'g3',
 'R' : 'g4'
}

with open('results_market.pickle', 'rb') as f:
    results_market = pickle.load(f)

with open('results_profits.pickle', 'rb') as f:
    results_profits = pickle.load(f)

# with open('results_market_gov.pickle', 'rb') as f:
#     results_market = pickle.load(f)
#
# with open('results_profits_gov.pickle', 'rb') as f:
#     results_profits = pickle.load(f)

rounds = results_market['Round'].max()

profits_calc = results_profits.loc[:, ~results_profits.columns.isin(['Round', 'Iteration'])]
market_calc = results_market.loc[:, ~results_market.columns.isin(['Round', 'Iteration'])]

results_profits['mean'] = profits_calc.mean(axis = 1)


for m in list(market_calc.stack().unique()):
    results_profits['mean_' + mg_dict[m]] = profits_calc[market_calc.eq(m)].mean(axis = 1)
    results_profits['count_' + mg_dict[m]] = market_calc[market_calc.eq(m)].count(axis = 1)

differences = 100 - (market_calc[1:].reset_index(drop=True) == market_calc[:-1].reset_index(drop=True)).sum(axis = 1)
differences.loc[results_profits['Iteration'] == 1000] = np.nan
results_profits['changes'] = differences

print(results_profits)

summary_columns = ['Iteration', 'mean', 'mean_g1', 'mean_g2', 'mean_g3', 'mean_g4', 'count_g1', 'count_g2', 'count_g3', 'count_g4', 'changes']
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



# plot1 - Num of agents vs Round
tmp = summary_data[['count_g1', 'count_g2', 'count_g3', 'count_g4']]
tmp.columns = ['market_1', 'market_2', 'market_3', 'market_4']

ax = tmp.plot(
    title = 'Average number of agents in the market by round \n (n_Agents = 100, n_Iterations = 100)',
    colormap = cm,
    kind = 'area',
    stacked = True,
    grid = True
)
ax.set_ylabel('Number of agents')
ax.set_xlabel('Round')
plt.savefig('art_Number_Iter.png', bbox_inches='tight', dpi = 300)
plt.show()

# plot2 - Profits vs Round
tmp = summary_data[['mean', 'mean_g1', 'mean_g2', 'mean_g3', 'mean_g4']]
tmp.columns = ['Total', 'market_1', 'market_2', 'market_3', 'market_4']

ax = tmp.plot(
    title = 'Average profit by round \n (n_Agents = 100, n_Iterations = 100)',
    colormap = cm,
    grid = True
)
for line in ax.get_lines():
    if line.get_label() == 'Total':
        line.set_linewidth(2)
        line.set_zorder(1)
    else:
        line.set_linewidth(1)
        line.set_zorder(0)

ax.set_ylabel('Profit')
ax.set_xlabel('Round')
plt.savefig('art_Profit_Iter.png', bbox_inches='tight', dpi = 300)
plt.show()

# #plot2 - Number of farmers in each market at the and of a simulation
# tmp  = pd.DataFrame({
# 'Market' : ['good_1', 'good_2', 'good_3', 'good_4', 'G'],
# 'count' : summary_data.loc[summary_data.index == 1000, ['count_g1', 'count_g2', 'count_g3', 'count_g4', 'count_G']].values.flatten().tolist(),
# 'profit' : summary_data.loc[summary_data.index == 1000, ['mean_g1', 'mean_g2', 'mean_g3', 'mean_g4', 'mean_G']].values.flatten().tolist()
#
# })
#
# ax = tmp.plot.bar(
#     title = 'Average number of agents in markets in 1000th Round',
#     legend = False,
#     x = 'Market',
#     y = 'count',
#     rot = 0,
#     colormap = cm
# )
# ax.set_xlabel('Market')
# ax.set_ylabel('count')
# for p in ax.patches:
#     ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
#
# plt.savefig('art_gov_Market_Farmers_bar.png', dpi = 200)
# plt.show()

#plot3 - Number of decision changes
ax = summary_data[['changes']].plot(
    title = 'Average decision changes by round \n (n_Agents = 100, n_Iterations = 100)',
    colormap = cm,
    grid = True
)
ax.set_ylabel("% of agents who changed their decision")
ax.set_xlabel('Round')
plt.savefig('art_Decision_chg.png', bbox_inches='tight', dpi = 300)
plt.show()

#plot4 - % of decision changes for each farmer vs its profits
def flatten(xss):
    return [x for xs in xss for x in xs]

import seaborn as sns
ax = sns.regplot(x=[x * 100 for x in flatten(agent_avg_changes)],
    y=flatten(agent_avg_profits),
    scatter_kws={'s':1, 'color' : '#24325F'},
    line_kws=dict(color = '#FB6467')
)
ax.set(title = 'Profits vs. % of changed decisions \n (each dot represents single agent in one experiment run)')
ax.set_ylabel('Profit')
ax.set_xlabel('% of changed decisions')
plt.savefig('art_Profits_chg.png', bbox_inches='tight', dpi = 300)
plt.show()



def plot_iter(iter, results_profits):
    round_data = results_profits.loc[results_profits['Round'] == iter].copy().reset_index()

    # plot1 - Num of agents vs Round
    tmp = round_data[['count_g1', 'count_g2', 'count_g3', 'count_g4']]
    tmp.columns = ['market_1', 'market_2', 'market_3', 'market_4']

    ax = tmp.plot(
        title = 'Number of agents in the market by round in iteration: ' + str(iter) + ' (n_Agents = 100)',
        colormap = cm,
        kind = 'area',
        stacked = True,
        grid = True
    )
    ax.set_ylabel('Number of agents')
    ax.set_xlabel('Round')
    plt.savefig('art_Num_Iter_' + str(iter) + '.png', bbox_inches='tight', dpi = 300)
    plt.show()

    #plot2 - Profits on market vs Round

    tmp = round_data[['mean', 'mean_g1', 'mean_g2', 'mean_g3', 'mean_g4']]
    tmp.columns = ['Total', 'market_1', 'market_2', 'market_3', 'market_4'
    ]

    ax = tmp.plot(
        title = 'Market profit by round in iteration: ' + str(iter) + ' (n_Agents = 100)',
        colormap = cm,
        grid = True
    )
    ax.set_ylabel('Profit')
    ax.set_xlabel('Round')
    for line in ax.get_lines():
        if line.get_label() == 'Total':
            line.set_linewidth(2)
            line.set_zorder(1)
        else:
            line.set_linewidth(1)
            line.set_zorder(0)

    plt.savefig('art_Profit_Iter_' + str(iter) + '.png', bbox_inches='tight', dpi = 300)
    plt.show()



    # tmp  = pd.DataFrame({
    # 'Market' : ['good_1', 'good_2', 'good_3', 'good_4', 'G'],
    # 'count' : round_data.loc[round_data.index == 1000, ['count_g1', 'count_g2', 'count_g3', 'count_g4', 'count_G']].values.flatten().tolist(),
    # 'profit' : round_data.loc[round_data.index == 1000, ['mean_g1', 'mean_g2', 'mean_g3', 'mean_g4', 'mean_G']].values.flatten().tolist()
    #
    # })
    #
    # ax = tmp.plot.bar(
    #     title = 'Number of agents in markets in 1000th iteration (round = ' + str(round) + ')',
    #     legend = False,
    #     x = 'Market',
    #     y = 'count',
    #     rot = 0,
    #     colormap = cm
    # )
    # ax.set_xlabel('Market')
    # ax.set_ylabel('count')
    # for p in ax.patches:
    #     ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    #
    # plt.savefig('art_gov_Market_Farmers_bar_' + str(round) + '.png', dpi = 200)
    # plt.show()


# max_idx = results_profits.loc[results_profits['Iteration'] == 1000, ['mean']].idxmax()
# min_idx = results_profits.loc[results_profits['Iteration'] == 1000, ['mean']].idxmin()
#
# max_round = results_profits.iloc[max_idx]['Round']
# min_round = results_profits.iloc[min_idx]['Round']
