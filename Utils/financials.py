''' Utility function with finance related tools '''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance  as yf

from inputs import START_DATE
from Utils.general import Plotter



### 1. Class related to data scrapping / query / loading
###############################################################################

class Prices:
    ''' class related to data query / extraction '''

    def __init__(self, ticker, start_date):
        self.data_web = self.get_data(ticker, start_date)
        self.data_normalized = self.normalize_data(self.data_web)


    def get_data(self, ticker, start_date):
        ''' download ticker close prices from, yahoo finance '''

        prices = yf.download(ticker, start=start_date, interval='1mo')['Close']
        prices.index = pd.to_datetime(prices.index)
        return prices


    def normalize_data(self, data):
        ''' normalize the given data, useful for macro indicators '''

        return (data - data.mean()) / data.std()



### 2. Classes related to financial metrics, i.e returns, ratios...
###############################################################################

class SerieStats:
    ''' Produce the financial metrics from a prices serie '''

    def __init__(self, prices_serie):
        self.data = prices_serie
        self.returns_serie = self.returns()
        self.sample_time = (self.data.index[-1] - self.data.index[0]).days


    def returns(self):
        ''' provides the simple returns of a prices serie '''

        ret = self.data / self.data.shift(1) - 1
        return ret.dropna()


    def total_returns(self):
        ''' provides the total return over the whole period '''

        return self.data.iloc[-1] / self.data.iloc[0] - 1


    def annualised_total_return(self):
        ''' provides the annualised total return over the whole period '''

        annual_return = self.total_returns()
        return (1 + annual_return)**(260 / self.sample_time) - 1


    def cumulative_returns(self):
        ''' provides the cumulative returns of a prices serie '''

        cum_returns = np.cumprod(1 + self.returns_serie)
        return np.log(cum_returns.dropna())


    def serie_std(self):
        ''' provides the std deviation of the serie of returns '''

        return self.returns_serie.std()


    def annualised_std(self):
        ''' provides the annualised std of the serie of returns
            considers only the price series to be daily or monthly '''

        std = self.serie_std()
        if np.busday_count(self.data.index[0].date(), self.data.index[1].date())>3:
            return std*np.sqrt(12)
        return std*np.sqrt(260)




class FinancialMetrics(SerieStats):
    ''' Produce the financial metrics from a prices serie '''

    def __init__(self, prices_serie):
        self.data = prices_serie
        super().__init__(self.data)


    def sharpe_ratio(self, risk_free_rate):
        ''' returns the sharpe ratio of the prices serie '''

        return (self.annualised_total_return() - risk_free_rate) / self.annualised_std()


    def sortino_ratio(self, risk_free_rate):
        ''' returns the sortino ratio of the prices serie '''

        numerator = self.annualised_total_return() - risk_free_rate
        downside_risk_std = self.returns_serie.loc[self.returns_serie < 0].std()
        if np.busday_count(self.data.index[0].date(), self.data.index[1].date())>3:
            return  numerator / (downside_risk_std*np.sqrt(12))
        return numerator / (downside_risk_std*np.sqrt(260))


    def maximum_drawdown(self):
        ''' returns a serie of drawdown of the prices serie. The minimum value of
            the returned serie is the maximum drawdown '''

        maximum_cumulative_returns = self.cumulative_returns().cummax()
        drawdown_serie = (self.cumulative_returns() / maximum_cumulative_returns) - 1
        return drawdown_serie


    def value_at_risk(self, level):
        ''' returns the VaR of the prices serie given a certain level '''

        return - np.percentile(self.returns_serie, 100*(1-level))


    def expected_shortfall(self, level):
        ''' returns the expected shortfall of the prices serie given a certain level '''

        value_at_risk = self.value_at_risk(level)
        tail_loss = self.returns_serie.loc[self.returns_serie < -value_at_risk]
        return - np.mean(tail_loss)



# beta
# alpha
# R-Squared
# information ratio

# contribution to return
# return explain with factor investing (size, value, momentum)
# investment repartition


# returns __________________________________________________________________

# cumulative returns vs benchmark choice
# table of monthly return with YTD at the end


# correlation _______________________________________________________________

# heatmap correlation of returns wrt asset classes






### 3. Classes related to weights computation
###############################################################################


class WeightsSetup:
    ''' Generic class with standard cleaning methods '''

    def __init__(self, weights, scores):
        self.weights = weights
        self.scores = scores


    def weights_cleaner(self, computed_weights):
        ''' performs the last cleaning before returning the computed weights '''

        cleaned_weights = computed_weights / np.sum(computed_weights, axis=1).values[:, np.newaxis] # to make sure the weights sum to 1
        na_values = self.scores.isnull().all(axis=1)
        cleaned_weights.loc[na_values] = cleaned_weights.loc[na_values].fillna(self.weights)
        return cleaned_weights


    def df_setup(self, row, weights, scores):
        ''' concat the dfs and prepare them for usage '''

        grading_df = pd.concat([weights.loc[row.name], scores.loc[row.name]], axis=1)
        grading_df.columns = ['benchmark weights','scores']
        grading_df.sort_values(by='scores', ascending=False, inplace=True)
        grading_df['cumul weights'] = np.cumsum(grading_df['benchmark weights'])
        return grading_df




class WeightTilting(WeightsSetup):
    ''' compute weights dependents on capitalization.
        needs dataframe of benchmark weight and df of scores as input '''

    def __init__(self, weights, scores):
        super().__init__(weights, scores)


    def __cap_weighting_util(self, row, weights, scores, pct_active):
        ''' utility function returning the assets weights at a row level '''

        grading_df = self.df_setup(row, weights, scores)
        pct_selected = max(grading_df.loc[grading_df['cumul weights'] <= pct_active,'cumul weights'])
        grading_df['total weights'] = grading_df['benchmark weights'] * (1 / pct_selected)
        grading_df.loc[grading_df['cumul weights'] > pct_selected,'total weights'] = 0
        return grading_df['total weights'].sort_index().fillna(0)


    def cap_weighting_tilt(self, pct_active):
        ''' gets a dataframe with weights in col 1 and cap in col 2, sorted best to worst
            & returns the df of tilted weights '''

        strat_weights = self.weights.copy(deep=True)
        strat_weights = strat_weights.apply(self.__cap_weighting_util, axis=1, args=(self.weights, self.scores, pct_active))
        return self.weights_cleaner(strat_weights)


    def cap_scaling_tilt(self):
        ''' gets a dataframe with weights in col 1 and cap in col 2, sorted best to worst
            & returns the df of tilted weights '''

        scores_strat = self.scores + 0.5 # to make it a scalable coef (increasing / decreasing) the bench weights
        weights_strat = self.weights * scores_strat
        return self.weights_cleaner(weights_strat)


    def signal_tilting(self, cutoff, max_underweight):
        ''' returns the df of signal tilted weights '''

        weights_tilt = np.ones(self.weights.shape)
        weights_tilt *= (self.scores > cutoff) + (self.scores <= cutoff)\
                        *(self.scores*max_underweight).where(self.scores*max_underweight > -self.weights, other=-self.weights)
        weights_tilt *= (self.scores <= cutoff) + (self.scores > cutoff)\
                        *(self.scores / np.sum(self.scores*(self.scores > cutoff), axis=1).values[:, np.newaxis])
        return self.weights_cleaner(weights_tilt)




class Reweighting(WeightsSetup):
    ''' compute weights independents from capitalization.
        needs dataframe of benchmark weight and score as input '''

    def __init__(self, weights, scores):
        super().__init__(weights, scores)


    def signal_weighting(self):
        ''' gives df of weights for reweighting signal weighting strategy '''

        computed_weights = self.scores / np.sum(self.scores, axis=1).values[:,np.newaxis]
        return self.weights_cleaner(computed_weights).dropna()


    def __equal_weighting_util(self, row, weights, scores, pct_active):
        ''' utility function returning the assets weights at a row level '''

        grading_df = self.df_setup(row, weights, scores)
        pct_selected = max(grading_df.loc[grading_df['cumul weights'] <= pct_active,'cumul weights'])
        grading_df['total weights'] = (1 / pct_selected) * (1 / grading_df.loc[grading_df['cumul weights'] <= pct_active,'cumul weights'].count())
        grading_df.loc[grading_df['cumul weights'] > pct_selected,'total weights'] = 0
        return grading_df['total weights'].sort_index().fillna(0)


    def equal_weighting(self, pct_active):
        ''' gives df of weights for reweighting equal weighting strategy '''

        strat_weights = self.weights.copy(deep=True)
        strat_weights = strat_weights.apply(self.__equal_weighting_util, axis=1, args=(self.weights, self.scores, pct_active))
        return self.weights_cleaner(strat_weights)




### 4. Classes related to backtesting
###############################################################################

class Backtest:
    ''' class dedicated to perf calc based on weights and asset prices '''

    def __init__(self, weights, assets_prices):
        self.weights = weights
        self.prices = assets_prices.loc[self.weights.index]
        self.wealth = pd.Series(10000 * np.ones(self.weights.shape[0]), index=self.weights.index)
        self.wealth_repartition = pd.DataFrame(np.ones(self.weights.shape), index=self.weights.index, columns=self.weights.columns)
        self.units_per_asset = self.wealth_repartition.copy(deep=True)


    def __order_columns(self):
        ''' set the df columns in alphabetical order '''

        self.wealth_repartition = self.wealth_repartition.reindex(sorted(self.wealth_repartition.columns), axis=1)
        self.units_per_asset = self.units_per_asset.reindex(sorted(self.units_per_asset), axis=1)


    def perf_back_office(self, row):
        ''' performs the low level operations for the perf calc method '''

        if row.name != self.units_per_asset.index[0]:
            previous_index = self.units_per_asset.index[self.units_per_asset.index.get_loc(row.name) - 1]
            self.wealth[row.name] = np.matmul(self.units_per_asset.loc[previous_index].T, self.prices.loc[row.name])
            self.wealth_repartition.loc[row.name] = self.weights.loc[row.name] * self.wealth[row.name]
            self.units_per_asset.loc[row.name] = self.wealth_repartition.loc[row.name] / self.prices.loc[row.name]
        else:
            pass
        return self.units_per_asset.loc[row.name]


    def perf_calc(self):
        ''' compute the wealth attributed to each asset over time '''

        self.__order_columns()
        self.wealth_repartition.iloc[0,:] = self.weights.iloc[0,:] * self.wealth[0]
        self.units_per_asset.iloc[0,:] = self.wealth_repartition.iloc[0,:] / self.prices.iloc[0,:]
        self.units_per_asset = self.units_per_asset.apply(self.perf_back_office, axis=1)
        return self.wealth




### 4. Classes related to streategy management
###############################################################################

class Strategy(Backtest, FinancialMetrics):
    ''' class to handle strategy with perf, returns... '''

    def __init__(self, weights, assets_prices):
        Backtest.__init__(self, weights, assets_prices)
        self.perf = self.perf_calc()
        FinancialMetrics.__init__(self, self.perf)






### 5. Classes related to macro analysis
###############################################################################


class VixAnalysis(Prices, Plotter):
    ''' analyses perf based on VIX '''

    ticker = ['^VIX']
    vix_data = Prices(ticker, START_DATE)

    def __init__(self, strat_perf):
        Plotter.__init__(self, strat_perf)
        self.perf = pd.DataFrame({'Strategy':strat_perf, 'VIX':VixAnalysis.vix_data.data_normalized, 'Nowcaster':strat_perf*0,
                                  'Cumul returns':self.cumulative_returns(strat_perf)})
        self.clustering()


    def clustering(self):
        ''' add cluster categories to the perf df based on indicator '''

        self.perf.loc[(self.perf['VIX'] < 0) & (self.perf['VIX'] < self.perf['VIX'].shift(1)), 'Nowcaster'] = 'Contraction'
        self.perf.loc[(self.perf['VIX'] < 0) & (self.perf['VIX'] > self.perf['VIX'].shift(1)), 'Nowcaster'] = 'Recovery'
        self.perf.loc[(self.perf['VIX'] > 0) & (self.perf['VIX'] < self.perf['VIX'].shift(1)), 'Nowcaster'] = 'Slowdown'
        self.perf.loc[(self.perf['VIX'] > 0) & (self.perf['VIX'] > self.perf['VIX'].shift(1)), 'Nowcaster'] = 'Expansion'
        self.perf.dropna(axis=0,how='any',inplace=True)


    def cumulative_returns(self, strat_perf):
        ''' provides the cumulative returns of a prices serie '''

        cum_returns = np.cumprod(strat_perf / strat_perf.shift(1))
        return np.log(cum_returns.dropna())


    def visual_comp(self):
        ''' plot the strategy returns clustered by VIX values '''

        sns.lineplot(data=self.perf, x=self.perf.index, y='Cumul returns', sort=False, color='gray', linewidth=1)
        sns.scatterplot(data=self.perf, x=self.perf.index, y='Cumul returns', hue='Nowcaster', palette='Set1')
        plt.title('Strategy Returns by VIX regime', fontsize=15)
        self.generic_end_commands()
































