import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genextreme as gev
from scipy.stats import probplot, laplace,t, norm
from .data_handler import DataHandler

COLORS = {
    'histogram': '#66FF66',
    'fitted_line': '#FF5733',
    'var_line': '#33FFCE',
    'shape': '#FF6F61',
    'location': '#A6FF33',
    'scale': '#3380FF'
    }


class GEVExtremeValueAnalysis(DataHandler):
    """

    """



    def __init__(self, symbol, start, end, alpha, block_size):
        super().__init__(symbol, start, end)
        self.alpha = alpha
        self.block_size = block_size

        # Pola do przechowywania wyników
        self.minima = None
        self.minima_indices = None
        self.block_boundaries = None
        self.num_minima = None

        self.shape = None
        self.loc = None
        self.scale = None
        self.var = None
        self.var_direct = None

        self.calculate_block_minima()
        if self.num_minima > 0:
            self.fit_gev()
        else:
            print("Brak minimów w danych. Nie można dopasować GEV.")

    def calculate_block_minima(self):
        """
        """
        minima = []
        minima_indices = []
        block_boundaries = []

        for i in range(0, len(self.returns), self.block_size):
            block = self.returns.iloc[i:i + self.block_size]
            if not block.empty:
                block_boundaries.append(block.index[0])
                min_value = block.min()
                minima.append(min_value)
                minima_indices.append(block.idxmin())

        if block_boundaries and block_boundaries[-1] != self.returns.index[-1]:
            block_boundaries.append(self.returns.index[-1])

        self.minima = minima
        self.minima_indices = minima_indices
        self.block_boundaries = block_boundaries
        self.num_minima = len(minima)

    def fit_gev(self):
        """
        """
        if self.minima is None or len(self.minima) == 0:
            print("No minma")
            return

        params = gev.fit(self.minima)
        self.shape, self.loc, self.scale = params

        self.var = gev.ppf(self.alpha, self.shape, loc=self.loc, scale=self.scale)

        self.var_direct = self.calculate_var_direct(
            self.shape, self.loc, self.scale, alpha=self.alpha
        )

    @staticmethod
    def calculate_var_direct(shape, loc, scale, alpha=0.01):
        """
        """
        xi = -shape
        if xi != 0:
            return loc + (scale / xi) * ((-np.log(alpha)) ** (-xi) - 1)
        else:
            return loc - scale * np.log(-np.log(alpha))

    def plot_minima(self):
        """
        """
        if self.minima is None or len(self.minima) == 0:
            print("No minima. Chart can't be plotted.")
            return

        minima_series = pd.Series(self.minima, index=self.minima_indices)

        plt.figure(figsize=(12, 6))
        plt.plot(self.returns, label='Log Returns', color='white', linewidth=1.0)

        plt.scatter(
            minima_series.index,
            minima_series.values,
            color='red',
            label='Block Minima',
            zorder=5,
            s=30
        )

        for i, x in enumerate(self.block_boundaries):
            plt.axvline(
                x,
                color='gray',
                linestyle='--',
                linewidth=0.8,
                label='Block Boundary' if i == 0 else ""
            )

        plt.title('Log Returns with Block Minima', color='white')
        plt.xlabel('Date', color='white')
        plt.ylabel('Log Returns', color='white')
        plt.tick_params(colors='white')
        plt.grid(False)
        plt.legend(facecolor='black', framealpha=0.8)
        plt.show()

    def plot_distribution(self):
        """
        Rysuje histogram block minima oraz dopasowaną dystrybucję GEV.
        Dodaje linię VaR.
        """
        if self.minima is None or self.shape is None:
            print("Brak dopasowanego GEV. Nie można narysować wykresu.")
            return

        x_min = min(self.minima)
        x_max = max(self.minima)
        var_line = self.var

        # Dostosowanie zakresu
        if var_line < x_min:
            x_min = var_line - 0.1 * abs(x_max - var_line)
        if var_line > x_max:
            x_max = var_line + 0.1 * abs(var_line - x_min)

        x = np.linspace(x_min, x_max, 300)
        fitted_gev_pdf = gev.pdf(x, self.shape, loc=self.loc, scale=self.scale)

        fig, ax = plt.subplots(figsize=(12, 8))
        counts, bins, _ = ax.hist(
            self.minima,
            bins=10,
            density=True,
            alpha=0.8,
            color=COLORS['histogram'],
            edgecolor='white',
            label='Block Minima Histogram'
        )

        ax.plot(x, fitted_gev_pdf, color=COLORS['fitted_line'], linewidth=2, label='Fitted GEV PDF')
        ax.axvline(var_line, color=COLORS['var_line'], linestyle='--', linewidth=2,
                   label=f'GEV VaR ({self.alpha*100:.1f}%)')

        ax.set_title('Fitted GEV Distribution with Block Minima', color='white')
        ax.set_xlabel('Block Minima Returns', color='white')
        ax.set_ylabel('Density', color='white')
        ax.set_ylim(0, max(counts) * 1.2 if len(counts) > 0 else 1)
        ax.tick_params(colors='white')
        ax.grid(color='gray', linestyle='--', linewidth=0.5)
        ax.legend()
        plt.show()

    def generate_results_table(self):
        """
        Zwraca DataFrame z wynikami: shape, loc, scale, liczba minimów, liczba obserwacji, VaR, VaR_direct, typ dystrybucji.
        """
        if self.minima is None or self.shape is None:
            print("Brak obliczonych parametrów. Upewnij się, że wczytano dane (init) i policzono minima/GEV.")
            return None

        distribution_type = self.determine_distribution(self.shape)

        df = pd.DataFrame({
            "Parameter": [
                "Shape (ξ)",
                "Location (μ)",
                "Scale (σ)",
                "Number of Minima",
                "Total Observations",
                f"VaR ({self.alpha*100}%)",
                f"Direct VaR ({self.alpha*100}%)",
                "Distribution Type"
            ],
            "Value": [
                self.shape,
                self.loc,
                self.scale,
                self.num_minima,
                len(self.returns),
                self.var,
                self.var_direct,
                distribution_type
            ]
        })

        return df

    @staticmethod
    def determine_distribution(shape):
        """
        """
        if shape > 0:
            return "Frechet (Heavy Tails)"
        elif shape == 0:
            return "Gumbel (Exponential Tails)"
        else:
            return "Weibull (Bounded Tails)"

    def summary(self):
        """
        """
        df = self.generate_results_table()
        if df is not None:
            print("\nExtreme Value Analysis Results:\n")
            print(df.to_string(index=False))

    def block_minima_by_target(self, returns, target_minima):
        """
        Dodatkowa metoda – oblicza block minima pod zadaną liczbę 'target_minima'.
        """
        block_size = len(returns) // target_minima
        minima_list = []
        for i in range(0, len(returns), block_size):
            block = returns.iloc[i:i + block_size]
            if not block.empty:
                minima_list.append(block.min())
        return minima_list, block_size

    def stability(self, target_minima_list):
        """
        Rysuje stabilność parametrów GEV (shape, loc, scale) oraz VaR 
        w funkcji liczby minimów w block approach (target_minima_list).
        """
        shape_params, loc_params, scale_params, var_values = [], [], [], []

        for target_minima in target_minima_list:
            minima_list, _ = self.block_minima_by_target(self.returns, target_minima)
            if len(minima_list) > 0:
                params = gev.fit(minima_list)
                shape, loc, scale = params
                var_val = gev.ppf(self.alpha, shape, loc=loc, scale=scale)
                shape_params.append(shape)
                loc_params.append(loc)
                scale_params.append(scale)
                var_values.append(var_val)
            else:
                shape_params.append(np.nan)
                loc_params.append(np.nan)
                scale_params.append(np.nan)
                var_values.append(np.nan)

        # Wykres parametrów
        plt.figure(figsize=(12, 6))
        plt.plot(target_minima_list, shape_params, marker='o', label='Shape (ξ)', color=COLORS['shape'])
        plt.plot(target_minima_list, loc_params, marker='x', label='Location (μ)', color=COLORS['location'])
        plt.plot(target_minima_list, scale_params, marker='^', label='Scale (σ)', color=COLORS['scale'])
        plt.title("Stability of GEV Parameters by Number of Minima", color='white')
        plt.xlabel("Number of Minima", color='white')
        plt.ylabel("Parameter Value", color='white')
        plt.tick_params(colors='white')
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.show()

        # Wykres VaR
        plt.figure(figsize=(12, 6))
        plt.plot(target_minima_list, var_values, marker='s', color='purple',
                 label=f'VaR ({self.alpha*100:.1f}%)')
        plt.title(f"Value at Risk (VaR) by Number of Minima at {self.alpha*100:.1f}% Significance Level", color='white')
        plt.xlabel("Number of Minima", color='white')
        plt.ylabel("VaR Value", color='white')
        plt.tick_params(colors='white')
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.show()

    def qq_plot(self):
        """
        Generate QQ plots for block minima compared to theoretical distributions:
        normal, Laplace, Student's t, the fitted GPD, and the fitted GEV.
        Distributions are fitted using all returns, but only minima are plotted.
        """
        if self.minima is None or len(self.minima) == 0:
            print("No minima data available – cannot generate QQ plot.")
            return

        # Fit distributions to all returns
        returns = self.returns.values
        mu, sigma = norm.fit(returns)  # Fit normal distribution
        loc_lap, scale_lap = laplace.fit(returns)  # Fit Laplace distribution
        df_t, loc_t, scale_t = t.fit(returns)  # Fit Student's t distribution

        # Fitted GEV parameters
        shape_gev, loc_gev, scale_gev = gev.fit(self.minima)

        # Extract minima values
        minima = self.minima

        # Create a 3x2 grid of QQ plots with black background
        fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharey=True, facecolor='black')
        fig.suptitle("QQ Plots for Block Minima", fontsize=16, color='white')

        # Normal distribution QQ plot
        probplot(minima, dist=norm, sparams=(mu, sigma), plot=axes[0, 0])
        axes[0, 0].set_title("Normal Distribution", color='white')
        axes[0, 0].grid(color='gray', linestyle='--', linewidth=0.7, alpha=0.5)

        # Laplace distribution QQ plot
        probplot(minima, dist=laplace, sparams=(loc_lap, scale_lap), plot=axes[0, 1])
        axes[0, 1].set_title("Laplace Distribution", color='white')
        axes[0, 1].grid(color='gray', linestyle='--', linewidth=0.7, alpha=0.5)

        # Student's t distribution QQ plot
        probplot(minima, dist=t, sparams=(df_t, loc_t, scale_t), plot=axes[1, 0])
        axes[1, 0].set_title("Student's t Distribution", color='white')
        axes[1, 0].grid(color='gray', linestyle='--', linewidth=0.7, alpha=0.5)

        # GEV distribution QQ plot (highlighted differently)
        probplot(minima, dist=gev, sparams=(shape_gev, loc_gev, scale_gev), plot=axes[1, 1])
        axes[1, 1].set_title("GEV Distribution", color='cyan')
        axes[1, 1].grid(color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
        axes[1, 1].set_facecolor('black')  # Highlight GEV plot background
        for spine in axes[1, 1].spines.values():
            spine.set_edgecolor('cyan')
            spine.set_linewidth(2)

        # Apply common axis settings
        for ax_row in axes:
            for ax in ax_row:
                ax.tick_params(colors='white')  # Set tick color to white for visibility
                ax.set_xlabel("Theoretical Quantiles", color='white')
                ax.set_ylabel("Sample Quantiles", color='white')

        # Adjust layout for better spacing
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()