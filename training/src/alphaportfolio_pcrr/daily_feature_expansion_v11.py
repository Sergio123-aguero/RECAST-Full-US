"""Market-relative features and rolling statistics."""
import numpy as np
import pandas as pd

STOCK_NAMES = ("market_beta_63", "market_beta_126", "idiosyncratic_volatility_63",
               "idiosyncratic_volatility_126", "residual_momentum_21",
               "residual_momentum_63", "market_correlation_63", "market_correlation_126")


def rolling(x, width, operation):
    r = pd.DataFrame(x).rolling(width, min_periods=width)
    return getattr(r, operation)().to_numpy(dtype=np.float64)
