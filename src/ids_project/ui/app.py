from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ids_project.contracts import NSL_KDD_COLUMNS, NUMERIC_COLUMNS
from ids_project.runtime import describe_runtime, load_runtime, predict_one
from ids_project.ui.data import (
    classification_frame,
    confusion_matrix_frame,
    kpi_cards,
    load_dashboard_sources,
    release_status,
    runtime_summary,
    simulation_samples,
    support_distribution_frame,
    top_features_frame,
)
from ids_project.ui.forms import (
    ADVANCED_FIELDS,
    DEFAULT_PAYLOAD,
    FLAG_OPTIONS,
    IMPORTANT_NUMERIC_FIELDS,
    PROTOCOL_OPTIONS,
    SERVICE_OPTIONS,
    build_payload,
)
from ids_project.ui.simulator_html import HTML_SIMULATOR_TEMPLATE
from ids_project.ui.style import APP_CSS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="streamlit run ids_project.ui.app")
    parser.add_argument("--artifact-dir", default="artifacts/final")
    parser.add_argument("--release-summary", default="reports/release/summary.json")
    parser.add_argument("--external-report", default="reports/external_validation/default-prod.json")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    st.set_page_config(
        page_title="IDS Network Detection",
        page_icon=":material/security:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)

    sources = _load_sources(args.release_summary, args.external_report, args.artifact_dir)
    page = _topbar()
    if page == "Dashboard":
        render_dashboard(sources)
    elif page == "Analyse":
        render_analysis(sources)
    elif page == "Tester":
        render_tester(sources)
    else:
        render_runtime(sources)


@st.cache_data(show_spinner=False)
def _load_sources(release_summary: str, external_report: str, artifact_dir: str):
    return load_dashboard_sources(
        release_summary_path=release_summary,
        external_report_path=external_report,
        artifact_dir=artifact_dir,
    )


@st.cache_resource(show_spinner=False)
def _load_runtime(artifact_dir: str):
    return load_runtime(artifact_dir)


def _topbar() -> str:
    pages = ["Dashboard", "Analyse", "Tester", "Runtime"]
    if "page" not in st.session_state:
        qp_page = st.query_params.get("page", "Dashboard")
        st.session_state["page"] = qp_page if qp_page in pages else "Dashboard"

    selected_page = st.session_state["page"]
    cols = st.columns(4)
    for col, page in zip(cols, pages):
        with col:
            if st.button(
                page,
                key=f"nav_top_{page}",
                use_container_width=True,
                type="primary" if page == selected_page else "secondary",
            ):
                st.session_state["page"] = page
                st.query_params["page"] = page
                st.rerun()
    return selected_page


def render_dashboard(sources) -> None:
    passed, failures = release_status(sources.release_summary)
    runtime = runtime_summary(sources.manifest)
    hero_profile = runtime.get("profile_name", "default-prod")
    hero_backend = runtime.get("metadata", {}).get("gpu_backend", "cpu")
    threshold = runtime.get("threshold", sources.external_report.get("threshold", "0.5"))

    st.markdown(
        f"<h1 class='ids-page-title' style='display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:0.4rem;'>"
        f"Reseau Security Analytics"
        f"<span class='ids-topline' style='margin-bottom:0;font-size:0.78rem;padding:2px 12px;'>Intrusion Detection System</span>"
        f"</h1>"
        f"<p class='ids-subtitle'>Profil {hero_profile} · backend {hero_backend} · seuil {threshold}</p>",
        unsafe_allow_html=True,
    )

    metric_help = {
        "Accuracy": "Proportion globale de predictions correctes sur le trafic observe.",
        "Macro F1": "Vue d'ensemble de la qualite de detection sur toutes les classes.",
        "Recall attaque": "Part des attaques effectivement detectees.",
        "R2L F1": "Performance sur les intrusions remote-to-local.",
        "U2R F1": "Performance sur les elevations de privilege.",
    }
    metric_defs = {
        "Accuracy": {"threshold": 0.70, "label": "Conforme"},
        "Macro F1": {"threshold": 0.50, "label": "Robuste"},
        "Recall attaque": {"threshold": 0.50, "label": "Securise"},
        "R2L F1": {"threshold": 0.30, "label": "Conforme"},
        "U2R F1": {"threshold": 0.10, "label": "Conforme"},
    }

    cards_html: list[str] = []
    for card in kpi_cards(sources.release_summary):
        label = card["label"]
        raw_value = card["raw"]
        value_text = card["value"]
        metric_def = metric_defs.get(label, {"threshold": 0.50, "label": "Conforme"})
        threshold_value = metric_def["threshold"]

        if raw_value is None:
            progress = 0.0
            status_color = "m3-color-gray"
            status_text = "m3-text-gray"
            footer = "N/A"
        else:
            progress = min(100.0, max(0.0, float(raw_value) * 100.0))
            is_valid = float(raw_value) >= threshold_value
            status_color = "m3-color-success" if is_valid else "m3-color-error"
            status_text = "m3-text-success" if is_valid else "m3-text-error"
            footer = metric_def["label"] if is_valid else "Hors-seuil"

        cards_html.append(
            f"""
            <div class="m3-metric-card" title="{metric_help.get(label, '')}">
              <div class="m3-metric-label-container">
                <span class="m3-metric-label">{label}</span>
                <span class="m3-metric-info-icon">i</span>
              </div>
              <div class="m3-metric-value">{value_text}</div>
              <div class="m3-metric-progress-container">
                <div class="m3-metric-progress-bg">
                  <div class="m3-metric-progress-fill {status_color}" style="width:{progress:.2f}%;"></div>
                </div>
                <div class="m3-metric-footer">
                  <span class="m3-metric-delta {status_text}">{footer}</span>
                  <span class="m3-metric-threshold">Seuil {threshold_value:.2f}</span>
                </div>
              </div>
            </div>
            """
        )

    components.html(
        f"""
        <style>{_metric_component_css()}</style>
        <div class="m3-metric-grid">{''.join(cards_html)}</div>
        """,
        height=175,
    )

    quality_class = "ids-pill" if passed else "ids-pill ids-pill-danger"
    quality_label = "Quality gates OK" if passed else "Quality gates a corriger"
    st.markdown(f"<div class='ids-status-row'><span class='{quality_class}'>{quality_label}</span></div>", unsafe_allow_html=True)
    if failures:
        _render_table(pd.DataFrame({"Controles echoues": failures}))

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown("#### Top features")
        top_features = top_features_frame(sources.external_report, limit=8)
        if top_features.empty:
            st.info("Top features indisponibles.")
        else:
            st.altair_chart(_bar_chart(top_features, "Feature", "Importance"), width="stretch")
    with right:
        st.markdown("#### Distribution classes")
        distribution = support_distribution_frame(sources.external_report, sources.manifest)
        if distribution.empty:
            st.info("Distribution indisponible.")
        else:
            st.altair_chart(_bar_chart(distribution, "Classe", "Support"), width="stretch")


def render_analysis(sources) -> None:
    st.markdown("<h2 class='ids-section-title'>Analyse des performances</h2>", unsafe_allow_html=True)
    if not sources.external_report:
        st.info("Le rapport externe est absent. Les visualisations ne sont pas disponibles.")
        return

    with st.expander("Comprendre les metriques", expanded=False):
        st.markdown(
            "- Precision : fiabilite des alertes.\n"
            "- Recall : capacite a ne pas laisser passer les attaques.\n"
            "- F1 : equilibre entre precision et recall.\n"
            "- Support : volume d'exemples evalues."
        )

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown("#### Matrice de confusion")
        confusion = confusion_matrix_frame(sources.external_report, sources.manifest)
        _render_table(confusion.reset_index(names="Classe"))
    with right:
        st.markdown("#### Repartition des classes")
        distribution = support_distribution_frame(sources.external_report, sources.manifest)
        if not distribution.empty:
            st.altair_chart(_bar_chart(distribution, "Classe", "Support"), width="stretch")

    metrics_col, features_col = st.columns([1, 1])
    with metrics_col:
        st.markdown("#### Metriques par classe")
        _render_table(classification_frame(sources.external_report, sources.manifest))
    with features_col:
        st.markdown("#### Features importantes")
        top_features = top_features_frame(sources.external_report)
        if not top_features.empty:
            st.altair_chart(_bar_chart(top_features, "Feature", "Importance"), width="stretch")


def _get_presets(manifest) -> dict[str, dict[str, Any]]:
    preferred_dataset = Path("data/raw/KDDTest+.txt")
    sample = simulation_samples(manifest, preferred_path=preferred_dataset)
    if sample.empty:
        sample = simulation_samples(manifest)

    presets: dict[str, dict[str, Any]] = {
        "normal_http": {
            **DEFAULT_PAYLOAD,
            "protocol_type": "tcp",
            "service": "http",
            "flag": "SF",
            "src_bytes": 220,
            "dst_bytes": 4500,
            "count": 2,
            "srv_count": 2,
            "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0,
            "label": "normal",
            "category": "normal",
            "name": "Trafic Web HTTP",
            "desc": "Navigation Web legitime sur un service sain.",
        },
        "normal_smtp": {
            **DEFAULT_PAYLOAD,
            "protocol_type": "tcp",
            "service": "smtp",
            "flag": "SF",
            "src_bytes": 520,
            "dst_bytes": 350,
            "count": 1,
            "srv_count": 1,
            "logged_in": 1,
            "label": "normal",
            "category": "normal",
            "name": "Session SMTP saine",
            "desc": "Envoi de mail legitime sans comportement suspect.",
        },
        "ddos_syn": {
            **DEFAULT_PAYLOAD,
            "protocol_type": "tcp",
            "service": "private",
            "flag": "S0",
            "count": 480,
            "srv_count": 24,
            "serror_rate": 1.0,
            "srv_serror_rate": 1.0,
            "same_srv_rate": 0.05,
            "diff_srv_rate": 0.07,
            "src_bytes": 0,
            "dst_bytes": 0,
            "label": "neptune",
            "category": "dos",
            "name": "TCP SYN Flood",
            "desc": "Saturation du service par ouverture TCP massive.",
        },
        "nmap_scan": {
            **DEFAULT_PAYLOAD,
            "protocol_type": "tcp",
            "service": "other",
            "flag": "REJ",
            "count": 1,
            "srv_count": 1,
            "same_srv_rate": 0.0,
            "diff_srv_rate": 1.0,
            "dst_host_count": 255,
            "dst_host_srv_count": 1,
            "dst_host_same_srv_rate": 0.0,
            "dst_host_diff_srv_rate": 1.0,
            "src_bytes": 0,
            "dst_bytes": 0,
            "label": "portsweep",
            "category": "probe",
            "name": "Scan de ports",
            "desc": "Balayage de ports pour cartographier la cible.",
        },
        "ping_death": {
            **DEFAULT_PAYLOAD,
            "protocol_type": "icmp",
            "service": "eco_i",
            "flag": "SF",
            "count": 200,
            "src_bytes": 65510,
            "dst_bytes": 0,
            "label": "pod",
            "category": "dos",
            "name": "Ping of Death",
            "desc": "Paquets ICMP geants et malformes.",
        },
        "sql_injection": {
            **DEFAULT_PAYLOAD,
            "protocol_type": "tcp",
            "service": "http",
            "flag": "SF",
            "count": 10,
            "srv_count": 1,
            "logged_in": 1,
            "num_compromised": 10,
            "serror_rate": 1.0,
            "srv_serror_rate": 1.0,
            "dst_host_srv_serror_rate": 1.0,
            "src_bytes": 1250,
            "dst_bytes": 8200,
            "label": "normal",
            "category": "r2l",
            "name": "Injection SQL",
            "desc": "Tentative d'extraction de donnees par requete applicative.",
        },
        "buffer_overflow": {
            **DEFAULT_PAYLOAD,
            "protocol_type": "tcp",
            "service": "telnet",
            "flag": "SF",
            "logged_in": 1,
            "root_shell": 1,
            "num_failed_logins": 3,
            "su_attempted": 1,
            "srv_rerror_rate": 1.0,
            "rerror_rate": 1.0,
            "src_bytes": 1820,
            "dst_bytes": 24800,
            "label": "buffer_overflow",
            "category": "u2r",
            "name": "Privilege escalation",
            "desc": "Exploit memoire visant un shell admin.",
        },
        "ssh_bruteforce": {
            **DEFAULT_PAYLOAD,
            "protocol_type": "tcp",
            "service": "other",
            "flag": "SF",
            "logged_in": 0,
            "num_failed_logins": 12,
            "count": 80,
            "srv_count": 1,
            "srv_rerror_rate": 1.0,
            "rerror_rate": 1.0,
            "dst_host_srv_rerror_rate": 1.0,
            "src_bytes": 0,
            "dst_bytes": 0,
            "label": "guess_passwd",
            "category": "r2l",
            "name": "SSH brute force",
            "desc": "Spraying de credentials sur un acces distant.",
        },
        "teardrop": {
            **DEFAULT_PAYLOAD,
            "protocol_type": "udp",
            "service": "private",
            "flag": "SF",
            "wrong_fragment": 3,
            "src_bytes": 28,
            "dst_bytes": 0,
            "label": "teardrop",
            "category": "dos",
            "name": "Teardrop fragment",
            "desc": "Fragments IP superposes pour destabiliser la pile.",
        },
        "backdoor": {
            **DEFAULT_PAYLOAD,
            "protocol_type": "tcp",
            "service": "private",
            "flag": "RSTR",
            "count": 200,
            "logged_in": 0,
            "rerror_rate": 1.0,
            "srv_rerror_rate": 1.0,
            "src_bytes": 160,
            "dst_bytes": 160,
            "label": "backdoor",
            "category": "r2l",
            "name": "Reverse shell backdoor",
            "desc": "Connexion persistante entrante ou sortante vers un attaquant.",
        },
    }

    if not sample.empty:
        for category in ("normal", "dos", "probe", "u2r", "r2l"):
            category_rows = sample[sample["category"] == category]
            if category_rows.empty:
                continue
            row = category_rows.iloc[0].to_dict()
            for preset in presets.values():
                if preset["category"] == category:
                    for key, value in row.items():
                        if key not in {"name", "desc"}:
                            preset[key] = value

    for key, preset in presets.items():
        preset["preset_key"] = key
    return presets


def _render_interactive_simulator(
    status: str,
    label: str,
    score: float,
    threshold: float,
    category: str,
    protocol: str = "tcp",
    service: str = "http",
    src_bytes: int = 180,
    count: int = 1,
    scenario: str = "idle",
) -> None:
    import time

    run_token = str(time.time_ns())
    html_content = (
        HTML_SIMULATOR_TEMPLATE.replace("__RUN_TOKEN__", run_token)
        .replace("__STATUS__", str(status))
        .replace("__LABEL__", str(label))
        .replace("__SCORE__", f"{float(score):.3f}")
        .replace("__THRESHOLD__", f"{float(threshold):.3f}")
        .replace("__CATEGORY__", str(category))
        .replace("__PROTOCOL__", str(protocol))
        .replace("__SERVICE__", str(service))
        .replace("__SRC_BYTES__", str(src_bytes))
        .replace("__COUNT__", str(count))
        .replace("__SCENARIO__", str(scenario))
    )
    components.html(html_content, height=330)


def render_tester(sources) -> None:
    st.markdown("<h2 class='ids-section-title'>Simuler & Tester le Reseau</h2>", unsafe_allow_html=True)
    if not sources.runtime_available:
        st.info("Les artefacts locaux sont absents. Le formulaire reste visible, mais la prediction est desactivee.")
        return

    presets = _get_presets(sources.manifest)
    scenario_groups = {
        "Flux legitimes": ["normal_http", "normal_smtp"],
        "Attaques reseau": ["ddos_syn", "teardrop", "nmap_scan", "ping_death"],
        "Exploits applicatifs": ["sql_injection", "ssh_bruteforce", "buffer_overflow", "backdoor"],
    }

    st.markdown("<div class='ids-topline'>Etape 1 : Choisir un scenario</div>", unsafe_allow_html=True)
    chooser_left, chooser_right = st.columns([0.42, 0.58])
    with chooser_left:
        selected_group = st.selectbox("Famille", list(scenario_groups.keys()), key="tester-group")
        selected_key = st.selectbox(
            "Scenario",
            scenario_groups[selected_group],
            key="tester-scenario",
            format_func=lambda key: presets[key]["name"],
        )
        selected_preset = presets[selected_key]
        st.markdown(
            f"""
            <div class="ids-card ids-reveal ids-reveal-delay-1">
              <div class="ids-scenario-kicker">{selected_group}</div>
              <h3 class="ids-scenario-title">{selected_preset['name']}</h3>
              <p class="ids-scenario-desc">{selected_preset['desc']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Lancer la simulation", key="run-selected-scenario", type="primary", use_container_width=True):
            st.session_state["active_payload"] = selected_preset
            st.session_state["simulation_source"] = "preset"
            st.rerun()
    with chooser_right:
        _render_scenario_summary(selected_preset)

    st.markdown("<div class='ids-topline'>Etape 2 : Simulation temps reel</div>", unsafe_allow_html=True)
    active_payload = st.session_state.get("active_payload")
    if active_payload is None:
        _render_interactive_simulator(
            status="idle",
            label="N/A",
            score=0.0,
            threshold=0.5,
            category="N/A",
            scenario="idle",
        )
        st.info("Choisissez un scenario et lancez la simulation, ou utilisez le formulaire manuel ci-dessous.")
    else:
        _render_simulation_result(sources, active_payload)

    with st.expander("Configuration manuelle avancee", expanded=False):
        values: dict[str, Any] = {}
        default_source = active_payload if active_payload is not None else DEFAULT_PAYLOAD
        with st.form("prediction-form"):
            cat_cols = st.columns(3)
            protocol_value = str(default_source.get("protocol_type", "tcp"))
            service_value = str(default_source.get("service", "http"))
            flag_value = str(default_source.get("flag", "SF"))
            values["protocol_type"] = cat_cols[0].selectbox(
                "Protocole",
                PROTOCOL_OPTIONS,
                index=PROTOCOL_OPTIONS.index(protocol_value) if protocol_value in PROTOCOL_OPTIONS else 0,
            )
            values["service"] = cat_cols[1].selectbox(
                "Service",
                SERVICE_OPTIONS,
                index=SERVICE_OPTIONS.index(service_value) if service_value in SERVICE_OPTIONS else 0,
            )
            values["flag"] = cat_cols[2].selectbox(
                "Flag",
                FLAG_OPTIONS,
                index=FLAG_OPTIONS.index(flag_value) if flag_value in FLAG_OPTIONS else 0,
            )

            st.markdown("#### Signaux principaux")
            for row in _chunked(IMPORTANT_NUMERIC_FIELDS, 4):
                columns = st.columns(len(row))
                for column, field in zip(columns, row):
                    values[field] = column.text_input(
                        field,
                        value=_format_numeric_value(default_source.get(field, DEFAULT_PAYLOAD[field])),
                        key=f"main-{field}",
                    )

            with st.expander("Champs avances"):
                for row in _chunked(ADVANCED_FIELDS, 4):
                    columns = st.columns(len(row))
                    for column, field in zip(columns, row):
                        values[field] = column.text_input(
                            field,
                            value=_format_numeric_value(default_source.get(field, DEFAULT_PAYLOAD[field])),
                            key=f"advanced-{field}",
                        )

            submitted = st.form_submit_button("Analyser ce flux", disabled=not sources.runtime_available)

        if submitted:
            try:
                payload = build_payload(_coerce_numeric_values(values))
                st.session_state["active_payload"] = payload
                st.session_state["simulation_source"] = "manual"
                st.rerun()
            except Exception as exc:
                st.error(f"Erreur de saisie : {exc}")


def _render_scenario_summary(preset: dict[str, Any]) -> None:
    summary = pd.DataFrame(
        [
            {"Signal": "protocol_type", "Valeur": str(preset.get("protocol_type", "tcp"))},
            {"Signal": "service", "Valeur": str(preset.get("service", "http"))},
            {"Signal": "flag", "Valeur": str(preset.get("flag", "SF"))},
            {"Signal": "src_bytes", "Valeur": str(preset.get("src_bytes", 0))},
            {"Signal": "dst_bytes", "Valeur": str(preset.get("dst_bytes", 0))},
            {"Signal": "count", "Valeur": str(preset.get("count", 0))},
            {"Signal": "srv_count", "Valeur": str(preset.get("srv_count", 0))},
        ]
    )
    st.markdown("<div class='ids-reveal ids-reveal-delay-2'>", unsafe_allow_html=True)
    _render_table(summary)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_simulation_result(sources, payload: dict[str, Any]) -> None:
    payload_clean = {key: value for key, value in payload.items() if key in NSL_KDD_COLUMNS}
    payload_kdd = build_payload(payload_clean)
    bundle = _load_runtime(str(sources.artifact_dir))
    result = predict_one(bundle, payload_kdd)

    status = "blocked" if result.label != "normal" else "normal"
    detection_method = "LightGBM"

    st.markdown("<div class='ids-reveal ids-reveal-delay-1'>", unsafe_allow_html=True)
    _render_interactive_simulator(
        status=status,
        label=result.label,
        score=result.score,
        threshold=result.threshold,
        category=result.category,
        protocol=str(payload.get("protocol_type", "tcp")),
        service=str(payload.get("service", "http")),
        src_bytes=int(float(payload.get("src_bytes", 0))),
        count=int(float(payload.get("count", 0))),
        scenario=str(payload.get("preset_key", "manual")),
    )
    st.markdown("</div>", unsafe_allow_html=True)

    badge_class = "ids-pill" if status == "normal" else "ids-pill ids-pill-danger"
    decision = "Connexion autorisee" if status == "normal" else "Attaquant bloque"
    details = pd.DataFrame(
        [
            {"Etape": "Type de flux", "Valeur": str(payload.get("name", "Flux manuel"))},
            {"Etape": "Categorie predite", "Valeur": result.category},
            {"Etape": "Label ML", "Valeur": result.label},
            {"Etape": "Methode", "Valeur": detection_method},
            {"Etape": "Score", "Valeur": f"{result.score:.3f}"},
            {"Etape": "Seuil", "Valeur": f"{result.threshold:.3f}"},
            {"Etape": "Action finale", "Valeur": "BLOCK" if status != "normal" else "ALLOW"},
        ]
    )
    st.markdown(
        f"""
        <div class="ids-card ids-reveal ids-result-delayed" style="margin-top: 1rem;">
          <div class="ids-result-head">
            <div>
              <h3 class="ids-result-title">{decision}</h3>
              <p class="ids-result-subtitle">Le moteur a analyse ce flux et produit une decision exploitable ci-dessous.</p>
            </div>
            <span class="{badge_class}">{"NORMAL" if status == "normal" else "BLOCKED"}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_table(details)


def render_runtime(sources) -> None:
    st.markdown("<h2 class='ids-section-title'>Runtime et artefacts</h2>", unsafe_allow_html=True)
    if not sources.manifest:
        st.info("Aucun manifest d'artefact local n'a ete trouve.")
    else:
        summary = runtime_summary(sources.manifest)
        runtime_rows = [
            {"Champ": "Modele", "Valeur": str(summary.get("model_name"))},
            {"Champ": "Profil", "Valeur": str(summary.get("profile_name"))},
            {"Champ": "Seuil", "Valeur": str(summary.get("threshold"))},
            {"Champ": "Features", "Valeur": str(summary.get("feature_count"))},
            {"Champ": "Dataset", "Valeur": str(summary.get("dataset_path"))},
        ]
        _render_table(pd.DataFrame(runtime_rows))
        mapping_rows = [{"Label": key, "Index": value} for key, value in summary.get("label_mapping", {}).items()]
        hash_rows = [{"Fichier": key, "Hash": value} for key, value in summary.get("artifact_hashes", {}).items()]
        left, right = st.columns(2)
        with left:
            st.markdown("#### Mapping labels")
            if mapping_rows:
                _render_table(pd.DataFrame(mapping_rows))
            else:
                st.info("Mapping indisponible.")
        with right:
            st.markdown("#### Integrite artefacts")
            if hash_rows:
                _render_table(pd.DataFrame(hash_rows))
            else:
                st.info("Hashes indisponibles.")

    st.warning("Les artefacts joblib doivent etre charges uniquement depuis une source locale de confiance.")
    if sources.runtime_available:
        bundle = _load_runtime(str(sources.artifact_dir))
        st.markdown("#### Runtime charge")
        runtime_details = describe_runtime(bundle)
        runtime_frame = pd.DataFrame(
            [{"Cle": key, "Valeur": str(value)} for key, value in runtime_details.items() if key != "feature_columns"]
        )
        _render_table(runtime_frame)
        with st.expander("Colonnes features runtime"):
            _render_table(pd.DataFrame({"Feature": runtime_details.get("feature_columns", [])}))


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _bar_chart(frame: pd.DataFrame, x_field: str, y_field: str) -> alt.Chart:
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#0b57d0")
        .encode(
            y=alt.Y(f"{x_field}:N", sort="-x", axis=alt.Axis(labelColor="#5e5e5e", title=None, labelLimit=200)),
            x=alt.X(f"{y_field}:Q", axis=alt.Axis(gridColor="#e1e2e9", labelColor="#5e5e5e", title=None)),
            tooltip=[x_field, y_field],
        )
        .properties(height=280)
        .configure_view(strokeWidth=0)
        .configure(background="#ffffff")
    )


def _render_table(frame: pd.DataFrame) -> None:
    html = frame.to_html(index=False, escape=True, classes="ids-table", border=0)
    st.markdown(f"<div class='ids-table-wrap'>{html}</div>", unsafe_allow_html=True)


def _format_numeric_value(value: object) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)


def _coerce_numeric_values(values: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(values)
    for column in NUMERIC_COLUMNS:
        if column in coerced:
            coerced[column] = float(str(coerced[column]).replace(",", "."))
    return coerced


def _metric_component_css() -> str:
    return """
    body {
      margin: 0;
      font-family: Inter, Arial, sans-serif;
      background: transparent;
      color: #1f1f1f;
    }
    .m3-metric-grid {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 16px;
      width: 100%;
      box-sizing: border-box;
    }
    .m3-metric-card {
      background: #ffffff;
      border: 1px solid #c4c7c5;
      border-radius: 16px;
      padding: 1.1rem;
      min-height: 128px;
      box-sizing: border-box;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .m3-metric-label-container {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .m3-metric-label {
      color: #5e5e5e;
      font-weight: 600;
      font-size: 0.82rem;
    }
    .m3-metric-info-icon {
      color: #5e5e5e;
      font-size: 0.74rem;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      border: 1px solid #c4c7c5;
      font-weight: 700;
    }
    .m3-metric-value {
      color: #1f1f1f;
      font-size: 2rem;
      font-weight: 800;
      line-height: 1;
    }
    .m3-metric-progress-container {
      margin-top: auto;
    }
    .m3-metric-progress-bg {
      width: 100%;
      height: 6px;
      background: #f0f4f9;
      border-radius: 99px;
      overflow: hidden;
      margin-bottom: 8px;
    }
    .m3-metric-progress-fill {
      height: 100%;
      border-radius: 99px;
      animation: metric-grow 700ms cubic-bezier(0.2, 0, 0, 1) both;
      transform-origin: left;
    }
    .m3-color-success { background: #146c2e; }
    .m3-color-error { background: #b3261e; }
    .m3-color-gray { background: #c4c7c5; }
    .m3-metric-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.72rem;
      font-weight: 600;
    }
    .m3-text-success { color: #146c2e; }
    .m3-text-error { color: #b3261e; }
    .m3-text-gray { color: #5e5e5e; }
    .m3-metric-threshold { color: #5e5e5e; font-weight: 500; }
    @keyframes metric-grow {
      from { transform: scaleX(0); }
      to { transform: scaleX(1); }
    }
    @media (max-width: 900px) {
      .m3-metric-grid { grid-template-columns: repeat(2, 1fr); }
    }
    """


if __name__ == "__main__":
    main(sys.argv[1:])
