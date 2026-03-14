"""
Compatibility shim for enhanced analysis functions.

The comprehensive evaluation module expects to import:

    from enhanced_analysis import ...

but the actual implementations live in ``evaluation.advanced_analysis``.
This module simply re-exports those functions so both import styles work.
"""

from evaluation.advanced_analysis import (  # noqa: F401
    plot_ideal_solar_curve,
    correlation_vif_analysis,
    calculate_weighted_score,
    hourly_error_analysis,
    iterative_learning,
    energy_market_impact_analysis,
)

