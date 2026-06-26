# Climate-risk extension to OnSSET

This branch adds an experimental climate-risk cost adjustment to the OnSSET electrification workflow. The aim is to allow settlement-level climate risk factors to influence least-cost technology selection by increasing technology-specific LCOE values where assets are exposed to climate hazards.

## Summary of changes

The original OnSSET workflow is extended in two places:

**onsset/onsset.py** has been modified to accept settlement-level risk-factor columns and include them in the LCOE calculation.
The scenario notebook has been modified to merge risk-factor inputs, run risk and no-risk scenarios, and compare resulting technology choices and LCOE values.

The risk factors are treated as cost-equivalent adjustments to the LCOE calculation. They are not added directly to the reported upfront investment-cost outputs. Instead, the model calculates a risk-cost equivalent from the relevant exposed asset cost, annualises this value over the risk annuity period, and adds it to the annual cost stream used to calculate LCOE.

This means that climate-risk exposure can make a technology less competitive in the least-cost comparison, while the standard investment-cost outputs continue to represent conventional technology investment rather than investment plus risk cost.

## Risk-factor columns

The model expects settlement-level risk-factor columns such as:

- rf_total_SA_PV
- rf_total_GRID_POLE
- rf_total_GRID_TRANSFORMER
- rf_total_MG_PV
- rf_total_MG_WIND

(Included but not currently used):
- rf_total_MG_PV_GENERATION
- rf_total_MG_WIND_GENERATION

These are merged into the OnSSET settlement dataframe using the settlement id.

For standalone PV, the scenario notebook can distinguish between ground-mounted and rooftop assumptions. If the input risk file contains separate columns such as `rf_total_SA_PV_ground` and `rf_total_SA_PV_roof`, the notebook selects one of these and maps it to the generic column `rf_total_SA_PV`, which is then used by **onsset.py**.

## Implementation in onsset.py

The `Technology.get_lcoe()` method has been extended with three optional penalty inputs:

- line_penalty (transmission and distribution line components, such as HV, MV, and LV lines)
- transformer_penalty (service transformers and substations)
- generation_penalty (the main technology asset or generation package, such as standalone PV capital cost)

For grid technologies, the transmission and distribution cost calculation is split into line costs, transformer/substation costs, connection costs, and powerhouse costs. The grid pole/line risk factor is applied only to the line-cost component, while the transformer risk factor is applied only to transformer and substation costs. These risk costs are kept separate from the base investment cost and are annualised before being included in the LCOE calculation.

A generation penalty refers to a risk factor applied to the capital cost of the generation asset or technology package. Currently, the `rf_total_GRID` column is not used in the grid LCOE calculation. For standalone PV, the standalone PV risk factor is applied to the calculated standalone PV capital investment. The resulting risk cost is annualised and added to annual costs before calculating LCOE. In simplified form:

- generation risk cost = generation capital investment × generation risk factor
- annual risk cost = generation risk cost / annuity factor
- LCOE increase = annual risk cost / annual electricity demand

This means that a higher standalone PV risk factor increases the standalone PV LCOE and may therefore change the least-cost technology assignment.

## Mini-grid PV and wind risk calculation

Mini-grid PV hybrid and mini-grid wind LCOEs are first calculated through the hybrid system optimisation. After the hybrid LCOE and investment cost have been calculated, the mini-grid risk factor is applied to the resulting mini-grid investment.

For mini-grid PV hybrid, the model calculates the hybrid LCOE and investment cost, then applies the `rf_total_MG_PV` risk factor to the hybrid investment. The resulting risk cost is annualised and converted into a USD/kWh increment using annual settlement electricity demand. This increment is then added to the `MG_PVHybrid` LCOE.

In simplified form:

`MG_PV risk cost = MG_PVHybrid investment × rf_total_MG_PV`
`annual MG_PV risk cost = MG_PV risk cost / annuity factor`
`MG_PVHybrid LCOE increase = annual MG_PV risk cost / annual electricity demand`

The same logic is applied to mini-grid wind using the `rf_total_MG_WIND` risk factor and the calculated mini-grid wind investment:

`MG_Wind risk cost = MG_Wind investment × rf_total_MG_WIND`
`annual MG_Wind risk cost = MG_Wind risk cost / annuity factor`
`MG_Wind LCOE increase = annual MG_Wind risk cost / annual electricity demand`

This approach allows the risk adjustment to affect the final technology comparison while preserving the original hybrid optimisation step. The model first estimates the least-cost hybrid system design, then adds an annualised risk-cost increment to the resulting mini-grid LCOE.

## Scenario structure

The scenario notebook is designed to run three comparable cases:

- No-risk baseline
- Risk-adjusted case with ground-mounted standalone PV
- Risk-adjusted case with rooftop standalone PV

The main toggles are:

`RUN_RISK = False`  # no-risk baseline

or:

`RUN_RISK = True`
`SA_PV_RISK_MODE = "ground"` or `"roof"`

Outputs are saved using distinct filenames, for example:

- SierraLeone_NoRisk_Results.csv
- SierraLeone_Risk_SAPV_ground_Results.csv
- SierraLeone_Risk_SAPV_roof_Results.csv

## Post-processing outputs

The notebook includes post-processing to compare risk-adjusted scenarios against the no-risk baseline. The main outputs include:

- mean and median LCOE change;
- LCOE change in USD/kWh and percent;
- number of settlements whose least-cost technology changes;
- population affected by technology reassignment;
- reassignment matrices showing flows between technologies, such as `SA_PV` → `Grid` or `MG_PVHybrid` → `SA_PV`;
- LCOE-change maps with both full-range and 5th–95th percentile capped colour scales.

The main comparisons are:

- ground-mounted `SA_PV` risk case vs no-risk baseline;
- rooftop `SA_PV` risk case vs no-risk baseline.

The roof-vs-ground comparison is used as a sensitivity or adaptation comparison.