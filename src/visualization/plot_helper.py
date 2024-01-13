import matplotlib.pyplot as plt
import numpy as np
def plot_capacity_degradation(battery_data,
                              figsize=(12, 12),
                              normalize=True,
                              title='',
                              n_legend_cols=3):
    num_colors = len(battery_data)
    colors = plt.cm.jet(np.linspace(0, 1, num_colors))
    plt.figure(figsize=figsize)
    for color, cell_data in zip(colors, battery_data):
        tag = cell_data.cell_id
        inner_plot_capacity_degradation(cell_data, normalize=normalize, color=color, label=f'{tag}')

    plt.grid()
    plt.title(title)
    plt.xlabel('Cycles')
    plt.ylabel('Capacity')
    # plt.ylim(0.8, 1.1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", ncol=n_legend_cols)


def inner_plot_capacity_degradation(cell_data,
                                    normalize=True,
                                    **kwargs):
    nominal_capacity = cell_data.nominal_capacity_in_Ah
    q_ds = [max(cycle.discharge_capacity_in_Ah) for cycle in cell_data.cycle_data]

    if normalize:
        q_ds = [ q_d/nominal_capacity for q_d in q_ds]
    x = np.arange(len(q_ds)) + 1
    plt.plot(x, q_ds, **kwargs)


def plot_cycle_key_feature(cycle_infos,
                           key_fea,
                           figsize=(18, 6),
                           title='',
                           cycle_start=None,
                           cycle_end=None,
                           index_start=None,
                           index_end=None,
                           n_legend_cols=3,
                           x_feature = 'time_in_s'):
    plt.figure(figsize=figsize)
    if key_fea == 'internal_resistance_in_ohm':
        y = [getattr(cycle_info, key_fea) for cycle_info in cycle_infos]
        x = [getattr(cycle_info, 'cycle_number') for cycle_info in cycle_infos]
        plt.plot(x[cycle_start:cycle_end], y[cycle_start:cycle_end])
        plt.xlabel('cycle')
    else:
        cycle_infos = cycle_infos[cycle_start:cycle_end]
        num_colors = len(cycle_infos)
        colors = plt.cm.jet(np.linspace(0, 1, num_colors))

        for color, cycle_info in zip(colors, cycle_infos):

            y = getattr(cycle_info, key_fea)
            if isinstance(y, np.float64):
                continue
            if x_feature and  hasattr(cycle_info, x_feature):
                x = getattr(cycle_info, x_feature)
            else:
                x = [i for i in range(len(y))]

            plt.plot(x[index_start:index_end], y[index_start:index_end],color=color, label=f'Cycle {cycle_info.cycle_number}')
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", ncol=n_legend_cols)
            plt.xlabel(x_feature)

    plt.grid()
    plt.ylabel(key_fea)
    plt.title(title)

# plt_result
import numpy as np
import matplotlib.pyplot as plt

def plot_result(y, y_pred):
    # Plot the results
    X=[i for i in range(len(y))]
    plt.scatter(X, y, label='Ground Truth', color='blue', alpha=0.5)
    plt.scatter(X, y_pred, label='Prediction', color='red')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()