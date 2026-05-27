from __future__ import annotations

import pandas as pd

from ids_project.ui.app import _hydrate_presets_from_validation
from ids_project.ui.forms import DEFAULT_PAYLOAD


def test_hydrate_presets_uses_exact_validation_labels_per_scenario():
    presets = {
        "ddos_syn": {**DEFAULT_PAYLOAD, "category": "dos", "name": "SYN", "desc": "SYN"},
        "teardrop": {**DEFAULT_PAYLOAD, "category": "dos", "name": "Teardrop", "desc": "Teardrop"},
        "nmap_scan": {**DEFAULT_PAYLOAD, "category": "probe", "name": "Scan", "desc": "Scan"},
    }
    sample = pd.DataFrame(
        [
            {**DEFAULT_PAYLOAD, "label": "neptune", "category": "dos", "count": 229},
            {**DEFAULT_PAYLOAD, "label": "teardrop", "category": "dos", "wrong_fragment": 3},
            {**DEFAULT_PAYLOAD, "label": "nmap", "category": "probe", "service": "private"},
        ]
    )

    _hydrate_presets_from_validation(presets, sample)

    assert presets["ddos_syn"]["validation_label"] == "neptune"
    assert presets["teardrop"]["validation_label"] == "teardrop"
    assert presets["nmap_scan"]["validation_label"] == "nmap"
    assert presets["teardrop"]["wrong_fragment"] == 3
