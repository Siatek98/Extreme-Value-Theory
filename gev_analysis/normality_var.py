import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t, laplace, kstest, shapiro, jarque_bera
from .data_handler import DataHandler

COLORS = {
    'histogram': '#66FF66',
    'fitted_line': '#FF5733',
    'var_line': '#33FFCE',
    'shape': '#FF6F61',
    'location': '#A6FF33',
    'scale': '#3380FF'
    }

class NormalityVaR(DataHandler):


    def __init__(self, symbol, start, end, alpha):
        super().__init__(symbol, start, end)
        self.alpha = alpha
        self.returns = self.returns.dropna() 
        self.mu, self.sigma = self.returns.mean(), self.returns.std(ddof=1)

    def calculate_var(self):
        """Calculate VaR based on empirical and normal distributions."""
        empirical_var = np.percentile(self.returns, self.alpha * 100)
        normal_var = norm.ppf(self.alpha, loc=self.mu, scale=self.sigma)
        return empirical_var, normal_var

    def normality_tests(self):
        """Conduct normality tests: Shapiro-Wilk and Jarque-Bera."""
        shapiro_stat, shapiro_p = shapiro(self.returns)
        jb_stat, jb_p = jarque_bera(self.returns)
        return shapiro_p, jb_p

    def plot_results(self):
        """Plot histogram with fitted normal distribution and VaR lines."""
        empirical_var, normal_var = self.calculate_var()
        fig, ax = plt.subplots(figsize=(12, 8))
        counts, bins, _ = ax.hist(
            self.returns,
            bins=50,
            density=True,
            alpha=0.7,
            color=COLORS['histogram'],
            edgecolor='white'
        )

        # Fitted normal PDF
        x = np.linspace(bins.min(), bins.max(), 500)
        ax.plot(
            x,
            norm.pdf(x, self.mu, self.sigma),
            color=COLORS['fitted_line'],
            linewidth=2,
            label='Fitted Normal Distribution'
        )
        ax.grid(color='gray', linestyle='--', linewidth=0.7, alpha=0.5)

        # VaR lines
        ax.axvline(
            empirical_var,
            color=COLORS['var_line'],
            linestyle='--',
            linewidth=2,
            label=f'Empirical VaR ({self.alpha*100}%)'
        )
        ax.axvline(
            normal_var,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Normal VaR ({self.alpha*100}%)'
        )

        # Add normality tests results
        shapiro_p, jb_p = self.normality_tests()
        textstr = f'Shapiro-Wilk p-value: {shapiro_p:.4f}\nJarque-Bera p-value: {jb_p:.4f}'
        props = dict(boxstyle='round', facecolor='black', edgecolor='white', alpha=0.7)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, va='top', bbox=props)

        ax.set_title(f'Empirical vs. Normal VaR ({self.alpha*100}%) - {self.symbol}', color='white')
        ax.set_xlabel('Log Returns', color='white')
        ax.set_ylabel('Density', color='white')
        ax.tick_params(colors='white')
        ax.legend()
        plt.show()

    def distribution_summary(self):
        """

        """

        df_t, loc_t, scale_t = t.fit(self.returns)
        # Laplace: (loc, scale)
        loc_lap, scale_lap = laplace.fit(self.returns)

        # VaR
        var_emp = np.percentile(self.returns, self.alpha * 100)
        var_norm = norm.ppf(self.alpha, loc=self.mu, scale=self.sigma)
        var_t    = t.ppf(self.alpha, df_t, loc=loc_t, scale=scale_t)
        var_lap  = laplace.ppf(self.alpha, loc=loc_lap, scale=scale_lap)


        # Normal
        _, ks_p_norm = kstest(self.returns, 'norm', args=(self.mu, self.sigma))
        # t-Student
        _, ks_p_t = kstest(self.returns, 't', args=(df_t, loc_t, scale_t))
        # Laplace
        _, ks_p_lap = kstest(self.returns, 'laplace', args=(loc_lap, scale_lap))

        data = [
            ['Normal', round(self.mu, 4), round(self.sigma, 4), round(ks_p_norm, 4), round(var_norm, 4)],
            ['T-Student', round(df_t, 4), round(scale_t, 4), round(ks_p_t, 4), round(var_t, 4)],
            ['Laplace', round(loc_lap, 4), round(scale_lap, 4), round(ks_p_lap, 4), round(var_lap, 4)],
            ['Empirical', None, None, None, round(var_emp, 4)]
        ]

        columns = ['Distribution', 'Param1', 'Param2', 'KS p-value', f'VaR {int(self.alpha*100)}%']
        df_summary = pd.DataFrame(data, columns=columns)

        return df_summary

    def plot_distributions(self):
        """

        """

        df_t, loc_t, scale_t = t.fit(self.returns)
        loc_lap, scale_lap = laplace.fit(self.returns)
        
        x = np.linspace(self.returns.min(), self.returns.max(), 500)

        pdf_norm = norm.pdf(x, loc=self.mu, scale=self.sigma)
        pdf_t    = t.pdf(x, df_t, loc=loc_t, scale=scale_t)
        pdf_lap  = laplace.pdf(x, loc=loc_lap, scale=scale_lap)


        fig, ax = plt.subplots(figsize=(12, 8))


        from scipy.stats import gaussian_kde
        kde = gaussian_kde(self.returns)
        pdf_emp = kde(x)
        ax.plot(x, pdf_emp, label='Empirical KDE', color='yellow', linewidth=2)


        ax.plot(x, pdf_norm, label='Normal PDF', color='blue', linewidth=2)
        ax.plot(x, pdf_t,    label='t-Student PDF', color='red', linewidth=2)
        ax.plot(x, pdf_lap,  label='Laplace PDF', color='green', linewidth=2)

        ax.legend()
        ax.grid(color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
        ax.set_title(f'Comparison of Empirical KDE vs. Theoretical PDFs - {self.symbol}', color='white')
        ax.set_xlabel('Log Returns', color='white')
        ax.set_ylabel('Density', color='white')
        ax.tick_params(colors='white')
        plt.show()
