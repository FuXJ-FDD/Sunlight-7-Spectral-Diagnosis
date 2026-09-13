from pathlib import Path
import numpy as np
import pandas as pd
from sunlight7.core import Parameters, load_sdr2025, compute_spectra, PHI_COLUMNS
from sunlight7.cluster import fit_baseline_gmm
import json

ROOT=Path(__file__).resolve().parents[1]

def test_scores_match_reference():
    p=Parameters.from_json(ROOT/'config'/'parameters.json')
    source=load_sdr2025(ROOT/'data'/'input'/'SDR2025_national_SDG_data.xlsx')
    calc=compute_spectra(source,p)
    ref=pd.read_excel(ROOT/'data'/'reference'/'reference_scores_and_archetypes.xlsx',sheet_name='Table S1 Results')
    m=calc.merge(ref,on='ISO3',suffixes=('_calc','_ref'))
    for c in ['Ωwhite']+PHI_COLUMNS:
        assert np.allclose(m[c+'_calc'],m[c+'_ref'],atol=1e-10,rtol=0)

def test_archetype_counts():
    cfg=json.loads((ROOT/'config'/'parameters.json').read_text())
    p=Parameters.from_json(ROOT/'config'/'parameters.json')
    source=load_sdr2025(ROOT/'data'/'input'/'SDR2025_national_SDG_data.xlsx')
    scores=compute_spectra(source,p)
    result=fit_baseline_gmm(scores,cfg['gmm'])
    assert result.assignments['Archetype'].value_counts().to_dict()=={
        'Environment–Governance Driven':68,
        'Multidimensional Laggards':57,
        'Balanced Achievers':31,
        'Economy–Innovation Driven':11,
    }
