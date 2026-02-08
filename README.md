# Multicurve SABR-FMM

This repository contains a Python implementation of a **Multicurve Forward Market Model (FMM)** with **SABR Stochastic Volatility** as per Lyashenko, Mercurio (2019). It features a pricer for **Bermudan Swaptions** using Least Squares Monte Carlo (LSM).

## Features

* **Multicurve Framework:** Explicit separation of OIS (discounting) and EUR-IBOR (forward projection) curves with deterministic basis spreads.

* **SABR Stochastic Volatility:** Each forward rate follows a (Normal) SABR process with $\beta = 0$. The model includes a calibration engine using Hagan's Normal Volatility expansion.

* **Drift :** Implements the drift adjustments required for FMM with stochastic volatility, as in Mercurio 2009.

* **Correlation Matrix:** Handles a $4N \times 4N$ matrix covering Rate-Rate, Vol-Vol, Skew, and Cross-Curve correlations. Supports PCA and parametric correlations (Exponential Decay or Two-Parameters).

* **Calibration** is performed against swaptions market normal volatility using Hagan's Normal Volatility approximation. The optimizer minimizes the Sum of Squared Errors (SSE) between model-implied and market volatilities for each tenor slice.
  
* **Monte Carlo Simulation:**
  * **Sobol Sequences:** For faster Quasi-Monte Carlo convergence.
  * **Brownian Bridges:** To construct effective path trajectories.
  * **Predictor-Corrector:** For time-dependent Drift adjustment.

* **Pricing and Greeks:**
    * **Bermudan Swaptions:** Prices using Longstaff-Schwartz (LSM) backward induction.
    * **Control Variates:** Uses analytical European swaption prices to reduce variance.
    * **Greeks:** Computes OIS Delta, EUR Delta, and Vega via pathwise bumping.

## Project Structure

```text
├── data/                       # Input CSV files (Yield curves, Vol surfaces)
├── logs/                       # Cached calibration parameters (JSON)
├── pics/                       # Saved plots
├── src/
│   ├── __init__.py
│   ├── calibration.py          # SABR calibration
│   ├── config.py               # Global configuration constants
│   ├── model.py                # MulticurveSABR_LMM class and Monte Carlo engine
│   ├── plotting.py             # Plots
│   ├── pricers.py              # LSM Bermudan pricer
│   └── utils.py                # Helpers
├── main.py                     # Entry point script
└── README.md
```

## Parameters
All model and simulation settings are located in src/config.py. 
You can modify::
* Simulation: SIM_PATHS (e.g., $2^{13}$), SIM_STEPS, RNG_TYPE ('sobol'/'standard').
* Physics: CORR_MODE ('pca', 'exp'), LMM_TENOR, RHO_OIS_EUR.
* Product: EXERCISE_DATES, STRIKE.

## Output
* Console: Valuation report including Bermudan price, European Control Variate adjustment, and Greeks (Delta/Vega).

* pics/: 3D plots of the Market vs. Model volatility surfaces and residual heatmaps (if CHECK_CALIBRATION = True in config).

* logs/: JSON files caching calibrated SABR parameters to speed up subsequent runs.


## References
* Andersen, L. (2000). _"A Simple Approach to the Pricing of Bermudan Swaptions in the Multi-Factor LIBOR Market Model."_ Journal of Computational Finance, 3(2), 5–32.
  
* Bartlett, B. (2006). "Hedged Monte Carlo: Low Variance Derivative Pricing with Objective Probabilities." Wilmott Magazine.
  
* Brigo, D., & Mercurio, F. (2006). _"Interest Rate Models - Theory and Practice: With Smile, Inflation and Credit."_ Springer Finance.

* Glasserman, P. (2004). _"Monte Carlo Methods in Financial Engineering."_ Springer.

* Hagan, P., Kumar, D., Lesniewski, A., & Woodward, D. (2002). _"Managing Smile Risk"_. Wilmott Magazine, 84–108.

* Hagan, P. S., & Lesniewski, A. (2009). _"LIBOR Market Model with SABR Style Stochastic Volatility."_ (Working Paper).

* Jäckel, P. (2002). _"Monte Carlo Methods in Finance."_ John Wiley & Sons.

* Longstaff, F. A., & Schwartz, E. S. (2001). _"Valuing American Options by Simulation: A Simple Least-Squares Approach."_ The Review of Financial Studies, 14(1), 113–147.

* Mercurio, F. (2009). _"LIBOR Market Models with Stochastic Volatility."_ Risk Magazine.

* Mercurio, F. (2010). _"Modern LIBOR Market Models: Using Different Curves for Projecting Rates and for Discounting."_ International Journal of Theoretical and Applied Finance.

* Mercurio, F. (2010). _"A Note on the Shifted-Lognormal Libor Market Model."_ Working Paper.
  
* Pallavicini, A., & Tarenghi, M. (2010). _"Interest-Rate Modelling with Multiple Yield Curves."_ International Journal of Theoretical and Applied Finance, 13(06), 871–896.

* Rebonato, R. (2002). _"Modern Pricing of Interest-Rate Derivatives: The LIBOR Market Model and Beyond."_ Princeton University Press.
