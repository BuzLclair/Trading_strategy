import numpy as np
import pandas as pd
import Utils.financials as u_fin
import Utils.general as u_gen

from inputs import ASSETS_TICKER, START_DATE


### 0. Data setup
###############################################################################

ASSETS_PRICES = u_fin.Prices(ASSETS_TICKER, START_DATE).data_web
assets_returns = u_fin.FinancialMetrics(ASSETS_PRICES).returns()




### 1. Benchmark (Equi-Weighting with rebalancing)
###############################################################################

benchmark = {}
benchmark['weights'] = pd.DataFrame((1/len(ASSETS_TICKER)) * np.ones(assets_returns.shape), index=assets_returns.index,
                                 columns=ASSETS_TICKER)

benchmark['strategy'] = u_fin.Strategy(benchmark['weights'], ASSETS_PRICES)
strategy_plot = u_gen.Plotter(benchmark['strategy'].cumulative_returns())

strategy_plot.single_plot('Equi-weight benchmark return')
# u_fin.VixAnalysis(benchmark['strategy'].perf_calc()).visual_comp()


