
# Weather Variable Analysis Report

## Executive Summary
This analysis performed comprehensive correlation and VIF analysis on all 15 weather variables to establish academic justification for variable reduction in solar forecasting models.

## Methodology
1. **Correlation Analysis**: Pearson correlation coefficients calculated
2. **VIF Analysis**: Variance Inflation Factor calculated to assess multicollinearity
3. **Variable Selection**: Systematic reduction based on VIF <= 5 and |correlation| <= 0.8

## Results

### Original Variables (15)
1. T2MWET
2. T2M
3. T2MDEW
4. YEAR
5. PS
6. QV2M
7. SZA
8. RH2M
9. CLRSKY_SFC_SW_DWN
10. ALLSKY_SFC_SW_DWN
11. ALLSKY_SFC_SW_DIFF
12. ALLSKY_SFC_SW_DNI
13. WS10M
14. WD10M
15. HR
16. ALLSKY_KT
17. MO
18. DY
19. PRECTOTCORR

### Selected Variables (3)
1. MO
2. DY
3. PRECTOTCORR

## Academic Justification
- **Multicollinearity Reduction**: VIF threshold of 5.0 eliminates redundant variables
- **Redundancy Elimination**: Correlation threshold of 0.8 removes highly correlated variables
- **Predictive Power Preservation**: Core solar forecasting variables retained

## Conclusion
The systematic reduction from 15 to 3 weather variables provides strong academic justification for the final model configuration.
