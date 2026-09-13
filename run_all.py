from __future__ import annotations

import argparse, json, platform, sys
from pathlib import Path
from importlib.metadata import version
import pandas as pd

from sunlight7.core import Parameters, load_sdr2025, compute_spectra, summary_statistics, parameter_table, formula_notes, PHI_COLUMNS
from sunlight7.cluster import fit_baseline_gmm, gmm_diagnostics, archetype_summary
from sunlight7.sensitivity import run_sensitivity
from sunlight7.plots import figure1, figure2, figure3, figure4, figure5, figures6_7, supplementary_figures, correlation_tables

ROOT=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser(description='Reproduce the Sunlight-7 SDR2025 analysis.')
    ap.add_argument('--quick',action='store_true',help='Use fewer GMM initializations for K=2–10 diagnostics.')
    ap.add_argument('--skip-figures',action='store_true')
    args=ap.parse_args()

    cfg=json.loads((ROOT/'config'/'parameters.json').read_text(encoding='utf-8'))
    p=Parameters.from_json(ROOT/'config'/'parameters.json')
    source=load_sdr2025(ROOT/'data'/'input'/'SDSN_SDGi_2026.xlsx')
    scores=compute_spectra(source,p)  # preserves source order for GMM reproducibility
    cluster=fit_baseline_gmm(scores,cfg['gmm'])
    assignments=cluster.assignments
    diagnostics=gmm_diagnostics(scores,cfg['gmm'],quick=args.quick)
    archetypes=archetype_summary(assignments)
    table_s5,fig_s4_source,scenario_country=run_sensitivity(source,scores,p,cluster)
    corr_r,corr_p=correlation_tables(scores)
    groups=pd.read_csv(ROOT/'config'/'country_groups.csv')

    out=ROOT/'outputs'; figs=out/'figures'; out.mkdir(exist_ok=True); figs.mkdir(exist_ok=True)
    # Table S1 sorted by nonlinear rank; assignment sheet remains in source order.
    table_s1=assignments.sort_values('Ω Rank').reset_index(drop=True)
    with pd.ExcelWriter(out/'Sunlight7_reproduction_results.xlsx',engine='openpyxl') as xw:
        table_s1.to_excel(xw,sheet_name='Table S1 Results',index=False)
        parameter_table(p).to_excel(xw,sheet_name='Parameters',index=False)
        summary_statistics(scores).to_excel(xw,sheet_name='Summary Stats',index=False)
        formula_notes().to_excel(xw,sheet_name='Formula Notes',index=False)
        assignments.to_excel(xw,sheet_name='Archetype Assignments',index=False)
        archetypes.to_excel(xw,sheet_name='Archetype Summary',index=False)
        diagnostics.to_excel(xw,sheet_name='GMM Diagnostics',index=False)
        table_s5.to_excel(xw,sheet_name='Table S5',index=False)
        fig_s4_source.to_excel(xw,sheet_name='Figure S4 Source Data',index=False)
        scenario_country.to_excel(xw,sheet_name='Scenario Country Results',index=False)
        corr_r.to_excel(xw,sheet_name='Pearson r matrix')
        corr_p.to_excel(xw,sheet_name='Pearson p matrix')
        scores[['ISO3','Country','Institutional input x4',PHI_COLUMNS[3]]].to_excel(xw,sheet_name='Figure S3 Data',index=False)
    table_s1.to_csv(out/'TableS1_country_scores_and_archetypes.csv',index=False)
    table_s5.to_csv(out/'TableS5_sensitivity_analysis.csv',index=False)

    if not args.skip_figures:
        shp=ROOT/'data'/'geodata'/'世界国家'/'世界国家.shp'
        figure1(table_s1,figs); figure2(assignments,figs,shp); figure3(assignments,figs); figure4(scores,figs); figure5(assignments,figs); figures6_7(assignments,groups,figs); supplementary_figures(scores,assignments,diagnostics,fig_s4_source,source,figs,shp,p.theta4)

    metadata={
        'python':sys.version,
        'platform':platform.platform(),
        **{pkg:version(pkg) for pkg in ['numpy','pandas','scipy','scikit-learn','matplotlib','openpyxl','geopandas']},
        'gmm':cfg['gmm'],
        'notes':'GMM input order is preserved from the SDR2025 source workbook. Archetype names are assigned from component mean profiles.'
    }
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(f'Completed. Outputs written to {out}')

if __name__=='__main__': main()
