
import matplotlib.pyplot as plt
import numpy as np

def plot_search_results(grid):
    """
    Params: 
        grid: A trained GridSearchCV object.
    """
    ## Results from grid search
    results = grid.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    means_train = results['mean_train_score']
    stds_train = results['std_train_score']

    ## Getting indexes of values per hyper-parameter
    masks=[]
    masks_names= list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_'+p_k].data==p_v))

    params=grid.param_grid

    ## Ploting results
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i+1:])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        y_2 = np.array(means_train[best_index])
        e_2 = np.array(stds_train[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
        ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='^',label='train' )
        ax[i].set_xlabel(p.upper())

    plt.legend()
    plt.show()

# plotting prediction line
def plot_powe_time(x_train,x_test,y_train,y_test, y_pred):
    fig3, ax1=plt.subplots()
    fig4, ax2=plt.subplots()
    plt.style.use('seaborn-darkgrid')
    ax1.set_xlabel("Speed (m/s)")
    ax2.plot(x_test.index, y_test, label='Test Values', c='darksalmon')
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Total (MW)")

    plt.style.use('seaborn-darkgrid')

    ax1.scatter(x_train['Speed'].values, y_train, alpha=0.85, label='Train set',c='darksalmon')
    ax1.scatter(x_test['Speed'].values, y_test, alpha=0.86, label='Test set', c='cornflowerblue')
    xs, ys = zip(*sorted(zip(x_test['Speed'].values, y_pred)))

    ax1.plot(xs,ys, linewidth=2.5,  label='Prediction',c='g')
    ax2.plot(x_test.index, y_pred,label='Predicted Values', c='mediumseagreen')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
            ncol=3, fancybox=True, shadow=True)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
            ncol=3, fancybox=True, shadow=True)
    plt.xticks(rotation=45)  # Rotates X-Axis Ticks by 45-degrees
    return fig3, fig4