# Extreme Value Theory (EVT) Analysis for Financial Risk

This repository provides Python code for analyzing financial risk through Extreme Value Theory (EVT), applied to market data. The primary focus is on Value at Risk (VaR) estimation using two EVT methods: Generalized Extreme Value (GEV) distribution and Generalized Pareto Distribution (GPD). The analysis includes a practical example using CD Projekt Red (CDR.WA) stock data. You can replace tickers, and other parameters freely to test your results.

---

## Features

### 1. Data Handling

- Download financial data using Yahoo Finance API.
- Compute logarithmic returns.

### 2. VaR Analysis

- Compare parametric and non-parametric VaR approaches.
- Visualize empirical distributions against theoretical distributions like Normal, t-Student, and Laplace.

### 3. EVT Implementation

- **GEV Analysis**: Extract block minima, fit GEV distribution, and compute VaR.
- **GPD Analysis**: Use Peak Over Threshold (POT) method, fit GPD distribution, and compute VaR.

### 4. Visualization

- Histograms with fitted distributions and VaR lines.
- Stability charts for parameters and VaR values.
- QQ plots to assess fit quality for extremes.

---

## Installation


1. Install repository:

```bash
pip install gev_analysis
```

---

## Example Usage

### Data Preparation

```python
from data_handler import DataHandler
handler = DataHandler(symbol="CDR.WA", start="2020-01-01", end="2021-03-31")
```

### Normality and VaR Analysis

```python
from normality_var import NormalityVaR
normality = NormalityVaR(symbol="CDR.WA", start="2020-01-01", end="2021-03-31", alpha=0.01)
normality.plot_results()  # Plot empirical and normal VaR comparison
normality.distribution_summary()  # Summary of fitted distributions
```

### GEV Analysis

```python
from gev_analysis import GEVExtremeValueAnalysis
gev_analysis = GEVExtremeValueAnalysis(symbol="CDR.WA", start="2020-01-01", end="2021-03-31", alpha=0.01, block_size=10)
gev_analysis.plot_minima()  # Visualize block minima
gev_analysis.stability(range(2, 50))  # Stability analysis for block size
gev_analysis.plot_distribution()
gev_analysis.summary()

```

### GPD Analysis

```python
from gpd_analysis import GPDExtremeValueAnalysis
gpd_analysis = GPDExtremeValueAnalysis(symbol="CDR.WA", start="2020-01-01", end="2021-03-31", alpha_input=0.01, threshold=-0.058) 
gpd_analysis.plot_minima()  # Visualize exceedances
gpd_analysis.stability(threshold_list=[-0.02, -0.04, -0.06])  # Stability analysis for thresholds
gpd_analysis.plot_distribution() # Histogram with fitted GPD PDF
gpd_analysis.summary()   
```

---

## Key Insights

1. **EVT Application**: EVT helps estimate probabilities and severities of rare events better than traditional methods.
2. **GEV vs. GPD**: Both methods have unique strengths. GPD is often more precise for VaR estimation due to its direct focus on exceedances.
3. **Parameter Selection**: Correct choice of block size (GEV) or threshold (GPD) is critical to accurate results.

---

## References

- **Presentation with examples**: [Extreme Value Theory Basics](https://github.com/Siatek98/Extreme-Value-Theory/blob/main/EVT_presentation.pdf)
- **Email for Queries**: [tomeksiat@gmail.com](mailto:tomeksiat@gmail.com)

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
