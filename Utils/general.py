import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns




### 1. Class related to linear regression
###############################################################################

def regress(y,x):
    ''' performs the regression of x over y '''

    expl_var = sm.add_constant(x)
    indep_var = y[y.index.isin(expl_var.index)]
    model = sm.OLS(indep_var, expl_var)
    return model.fit()



### 2. Class related to graphic plotting
###############################################################################

class Plotter:
    ''' class related to plot, init with one curve, others can be added
        afterwards when calling multi-plotting methods '''

    def __init__(self, curve):
        self.curve = curve
        self.generic_setup()


    def generic_setup(self):
        ''' set up the generic type layout '''

        sns.set(rc = {"figure.figsize":(10,5)})
        sns.set_style({'grid.color':'lightgrey','grid.linestyle': ':',
                       'axes.facecolor': 'whitesmoke', 'axes.linewidth': 1,
                       'axes.edgecolor':'dimgrey'})


    def generic_end_commands(self):
        ''' run the last usual commands in a plot '''

        plt.gca().xaxis.grid(False)
        plt.gca().spines[['top','right']].set_color('none')
        plt.tight_layout()
        plt.show()


    def single_plot(self, title=None):
        ''' plot the curves given in init only '''

        sns.lineplot(self.curve, linewidth=1)
        plt.title(title, fontsize=15)
        self.generic_end_commands()


    def multi_plot(self, curves, title=None, legends=None):
        ''' plot the the additional curves given (curves var must be a list) '''

        plt.plot(self.curve, linewidth=1)
        for curve in curves:
            plt.plot(curve, linewidth=1)
        plt.title(title, fontsize=15)
        plt.legend(legends)
        self.generic_end_commands()














