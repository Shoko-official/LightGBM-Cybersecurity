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
from ids_project.runtime import describe_runtime, load_runtime, predict_batch, predict_one
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

SCENARIO_LABEL_CANDIDATES = {
    "normal_http": ("normal",),
    "normal_smtp": ("normal",),
    "ddos_syn": ("neptune", "smurf", "apache2", "mailbomb"),
    "nmap_scan": ("nmap", "portsweep", "mscan", "satan", "saint"),
    "ping_death": ("pod",),
    "teardrop": ("teardrop",),
    "sql_injection": ("sqlattack", "phf", "warezmaster"),
    "buffer_overflow": ("buffer_overflow", "loadmodule", "perl", "rootkit", "ps", "xterm"),
    "ssh_bruteforce": ("guess_passwd", "snmpguess", "warezmaster"),
    "backdoor": ("backdoor", "httptunnel", "multihop", "warezmaster"),
}

T = {
    "en": {
        "nav_dashboard": "Dashboard",
        "nav_analyse": "Analysis",
        "nav_tester": "Tester",
        "nav_runtime": "Runtime",

        "page_title": "Network Security Analytics",
        "page_subtitle": "Profile {profile} · backend {backend} · threshold {threshold}",

        "metric_accuracy": "Accuracy",
        "metric_macro_f1": "Macro F1",
        "metric_recall_attack": "Attack Recall",
        "metric_r2l_f1": "R2L F1",
        "metric_u2r_f1": "U2R F1",

        "help_accuracy": "Overall proportion of correct predictions on observed traffic.",
        "help_macro_f1": "Overall quality of detection across all classes.",
        "help_recall_attack": "Proportion of actual attacks successfully detected.",
        "help_r2l_f1": "Performance on remote-to-local intrusions.",
        "help_u2r_f1": "Performance on user-to-root privilege escalations.",

        "metric_compliant": "Compliant",
        "metric_robust": "Robust",
        "metric_secure": "Secure",
        "metric_below_threshold": "Below threshold",
        "metric_threshold": "Threshold {val:.2f}",

        "chart_top_features": "Top Features",
        "chart_distribution": "Class Distribution",
        "info_unavailable": "Unavailable.",

        "analysis_section_title": "Performance Analysis",
        "analysis_help_title": "Understanding Metrics",
        "analysis_help_content": "- **Precision**: reliability of alerts.\n- **Recall**: ability to detect all attacks.\n- **F1**: balance between precision and recall.\n- **Support**: volume of evaluated examples.",
        "analysis_confusion_matrix": "Confusion Matrix",
        "analysis_class_distribution": "Class Distribution",
        "analysis_metrics_class": "Metrics by Class",
        "analysis_top_features": "Important Features",

        "tester_section_title": "Simulate & Test Network",
        "tester_not_available": "Local artifacts are missing. The manual form remains visible, but prediction is disabled.",
        "tester_waiting_info": "Choose a scenario and start the simulation, or use the manual form below.",
        "tester_manual_title": "Advanced Manual Configuration",
        "tester_protocol": "Protocol",
        "tester_service": "Service",
        "tester_flag": "Flag",
        "tester_main_signals": "Main Signals",
        "tester_advanced_fields": "Advanced Fields",
        "tester_analyze_button": "Analyze this Flow",
        "tester_input_error": "Input error: {exc}",
        "tester_final_action": "Final Action",
        "tester_family": "Family",
        "tester_scenario": "Scenario",
        "tester_start_simulation": "Start Simulation",
        "tester_manual_flow": "Manual Flow",

        "decision_allowed": "Connection Allowed",
        "decision_blocked": "Attacker Blocked",
        "decision_subtitle": "The engine analyzed this flow and produced the decision below.",

        "step_flow_type": "Flow Type",
        "step_validation_label": "Validation Label",
        "step_predicted_category": "Predicted Category",
        "step_ml_label": "ML Label",
        "step_method": "Method",
        "step_score": "Score",
        "step_threshold": "Threshold",

        "runtime_section_title": "Runtime & Artifacts",
        "runtime_no_manifest": "No local artifact manifest was found.",
        "runtime_warning": "Joblib artifacts should only be loaded from a trusted local source.",
        "runtime_loaded_title": "Loaded Runtime",
        "runtime_feature_columns": "Runtime Feature Columns",
        "runtime_label_mapping": "Label Mapping",
        "runtime_artifact_integrity": "Artifact Integrity",

        "col_class": "Class",
        "col_precision": "Precision",
        "col_recall": "Recall",
        "col_f1": "F1",
        "col_support": "Support",
        "col_feature": "Feature",
        "col_importance": "Importance",
        "col_field": "Field",
        "col_value": "Value",
        "col_file": "File",
        "col_hash": "Hash",
        "col_key": "Key",
        "col_step": "Step",
    },
    "fr": {
        "nav_dashboard": "Tableau de Bord",
        "nav_analyse": "Analyse",
        "nav_tester": "Tester",
        "nav_runtime": "Runtime",

        "page_title": "Analyses de Sécurité Réseau",
        "page_subtitle": "Profil {profile} · backend {backend} · seuil {threshold}",

        "metric_accuracy": "Précision Globale",
        "metric_macro_f1": "F1 Macro",
        "metric_recall_attack": "Rappel d'Attaque",
        "metric_r2l_f1": "F1 R2L",
        "metric_u2r_f1": "F1 U2R",

        "help_accuracy": "Proportion globale de prédictions correctes sur le trafic observé.",
        "help_macro_f1": "Vue d'ensemble de la qualité de détection sur toutes les classes.",
        "help_recall_attack": "Part des attaques effectivement détectées.",
        "help_r2l_f1": "Performance sur les intrusions remote-to-local (R2L).",
        "help_u2r_f1": "Performance sur les élévations de privilèges (U2R).",

        "metric_compliant": "Conforme",
        "metric_robust": "Robuste",
        "metric_secure": "Sécurisé",
        "metric_below_threshold": "Hors-seuil",
        "metric_threshold": "Seuil {val:.2f}",

        "chart_top_features": "Caractéristiques Principales",
        "chart_distribution": "Distribution des Classes",
        "info_unavailable": "Indisponible.",

        "analysis_section_title": "Analyse des Performances",
        "analysis_help_title": "Comprendre les Métriques",
        "analysis_help_content": "- **Précision** : fiabilité des alertes.\n- **Rappel** : capacité à ne pas laisser passer les attaques.\n- **F1** : équilibre entre précision et rappel.\n- **Support** : volume d'exemples évalués.",
        "analysis_confusion_matrix": "Matrice de Confusion",
        "analysis_class_distribution": "Répartition des Classes",
        "analysis_metrics_class": "Métriques par Classe",
        "analysis_top_features": "Caractéristiques Importantes",

        "tester_section_title": "Simuler & Tester le Réseau",
        "tester_not_available": "Les artefacts locaux sont absents. Le formulaire reste visible, mais la prédiction est désactivée.",
        "tester_waiting_info": "Choisissez un scénario et lancez la simulation, ou utilisez le formulaire manuel ci-dessous.",
        "tester_manual_title": "Configuration Manuelle Avancée",
        "tester_protocol": "Protocole",
        "tester_service": "Service",
        "tester_flag": "Flag",
        "tester_main_signals": "Signaux Principaux",
        "tester_advanced_fields": "Champs Avancés",
        "tester_analyze_button": "Analyser ce Flux",
        "tester_input_error": "Erreur de saisie : {exc}",
        "tester_final_action": "Action finale",
        "tester_family": "Famille",
        "tester_scenario": "Scénario",
        "tester_start_simulation": "Lancer la simulation",
        "tester_manual_flow": "Flux manuel",

        "decision_allowed": "Connexion Autorisée",
        "decision_blocked": "Attaquant Bloqué",
        "decision_subtitle": "Le moteur a analysé ce flux et produit une décision exploitable ci-dessous.",

        "step_flow_type": "Type de flux",
        "step_validation_label": "Label de validation",
        "step_predicted_category": "Catégorie prédite",
        "step_ml_label": "Label ML",
        "step_method": "Méthode",
        "step_score": "Score",
        "step_threshold": "Seuil",

        "runtime_section_title": "Runtime et Artefacts",
        "runtime_no_manifest": "Aucun manifest d'artefact local n'a été trouvé.",
        "runtime_warning": "Les artefacts joblib doivent être chargés uniquement depuis une source locale de confiance.",
        "runtime_loaded_title": "Runtime Chargé",
        "runtime_feature_columns": "Colonnes Features Runtime",
        "runtime_label_mapping": "Mapping des Labels",
        "runtime_artifact_integrity": "Intégrité des Artefacts",

        "col_class": "Classe",
        "col_precision": "Précision",
        "col_recall": "Rappel",
        "col_f1": "F1",
        "col_support": "Support",
        "col_feature": "Caractéristique",
        "col_importance": "Importance",
        "col_field": "Champ",
        "col_value": "Valeur",
        "col_file": "Fichier",
        "col_hash": "Hash",
        "col_key": "Clé",
        "col_step": "Étape",
    }
}


def get_user_language() -> str:
    # 1. Query parameters
    try:
        query_lang = st.query_params.get("lang")
        if query_lang:
            if str(query_lang).lower().startswith("fr"):
                return "fr"
            if str(query_lang).lower().startswith("en"):
                return "en"
    except Exception:
        pass

    # 2. Accept-Language header
    try:
        headers = st.context.headers
        accept_language = headers.get("accept-language", "")
        if accept_language:
            first = accept_language.split(",")[0].strip().lower()
            if first.startswith("fr"):
                return "fr"
            if first.startswith("en"):
                return "en"
    except Exception:
        pass

    # 3. System locale
    try:
        import locale
        sys_lang, _ = locale.getlocale()
        if not sys_lang:
            sys_lang, _ = locale.getdefaultlocale()
        if sys_lang and sys_lang.lower().startswith("fr"):
            return "fr"
    except Exception:
        pass

    return "en"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="streamlit run ids_project.ui.app")
    parser.add_argument("--artifact-dir", default="artifacts/latest")
    parser.add_argument("--release-summary", default="reports/release/summary.json")
    parser.add_argument("--external-report", default="reports/latest/validation_report.json")
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

    if "lang" not in st.session_state:
        st.session_state["lang"] = get_user_language()

    with st.sidebar:
        st.markdown("### Language / Langue")
        lang_options = {"en": "English", "fr": "Français"}
        selected_lang = st.selectbox(
            "Select Language / Sélectionner la Langue",
            options=list(lang_options.keys()),
            format_func=lambda x: lang_options[x],
            index=list(lang_options.keys()).index(st.session_state["lang"]),
            key="sidebar_lang_selector"
        )
        if selected_lang != st.session_state["lang"]:
            st.session_state["lang"] = selected_lang
            st.rerun()

    lang = st.session_state["lang"]
    sources = _load_sources(args.release_summary, args.external_report, args.artifact_dir)
    page = _topbar(lang)
    if page == "Dashboard":
        render_dashboard(sources, lang)
    elif page == "Analyse":
        render_analysis(sources, lang)
    elif page == "Tester":
        render_tester(sources, lang)
    else:
        render_runtime(sources, lang)


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


def _topbar(lang: str) -> str:
    pages_internal = ["Dashboard", "Analyse", "Tester", "Runtime"]
    pages_display = {
        "Dashboard": T[lang]["nav_dashboard"],
        "Analyse": T[lang]["nav_analyse"],
        "Tester": T[lang]["nav_tester"],
        "Runtime": T[lang]["nav_runtime"]
    }
    if "page" not in st.session_state:
        qp_page = st.query_params.get("page", "Dashboard")
        st.session_state["page"] = qp_page if qp_page in pages_internal else "Dashboard"

    selected_page = st.session_state["page"]
    cols = st.columns(4)
    for col, page in zip(cols, pages_internal):
        with col:
            if st.button(
                pages_display[page],
                key=f"nav_top_{page}",
                use_container_width=True,
                type="primary" if page == selected_page else "secondary",
            ):
                st.session_state["page"] = page
                st.query_params["page"] = page
                st.rerun()
    return selected_page


def render_dashboard(sources, lang: str) -> None:
    passed, failures = release_status(sources.release_summary)
    runtime = runtime_summary(sources.manifest)
    hero_profile = runtime.get("profile_name", "default-prod")
    hero_backend = runtime.get("metadata", {}).get("gpu_backend", "cpu")
    threshold = runtime.get("threshold", sources.external_report.get("threshold", "0.5"))

    title_text = T[lang]["page_title"]
    subtitle_text = T[lang]["page_subtitle"].format(profile=hero_profile, backend=hero_backend, threshold=threshold)

    st.markdown(
        f"<h1 class='ids-page-title' style='display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:0.4rem;'>"
        f"{title_text}"
        f"</h1>"
        f"<p class='ids-subtitle'>{subtitle_text}</p>",
        unsafe_allow_html=True,
    )

    metric_labels_trans = {
        "Accuracy": T[lang]["metric_accuracy"],
        "Macro F1": T[lang]["metric_macro_f1"],
        "Recall attaque": T[lang]["metric_recall_attack"],
        "R2L F1": T[lang]["metric_r2l_f1"],
        "U2R F1": T[lang]["metric_u2r_f1"],
    }

    metric_help = {
        T[lang]["metric_accuracy"]: T[lang]["help_accuracy"],
        T[lang]["metric_macro_f1"]: T[lang]["help_macro_f1"],
        T[lang]["metric_recall_attack"]: T[lang]["help_recall_attack"],
        T[lang]["metric_r2l_f1"]: T[lang]["help_r2l_f1"],
        T[lang]["metric_u2r_f1"]: T[lang]["help_u2r_f1"],
    }

    metric_defs = {
        "Accuracy": {"threshold": 0.70, "label": T[lang]["metric_compliant"]},
        "Macro F1": {"threshold": 0.50, "label": T[lang]["metric_robust"]},
        "Recall attaque": {"threshold": 0.50, "label": T[lang]["metric_secure"]},
        "R2L F1": {"threshold": 0.30, "label": T[lang]["metric_compliant"]},
        "U2R F1": {"threshold": 0.10, "label": T[lang]["metric_compliant"]},
    }

    cards_html: list[str] = []
    for card in kpi_cards(sources.external_report):
        raw_label = card["label"]
        label = metric_labels_trans.get(raw_label, raw_label)
        raw_value = card["raw"]
        value_text = card["value"]
        metric_def = metric_defs.get(raw_label, {"threshold": 0.50, "label": T[lang]["metric_compliant"]})
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
            footer = metric_def["label"] if is_valid else T[lang]["metric_below_threshold"]

        t_threshold = T[lang]["metric_threshold"].format(val=threshold_value)
        cards_html.append(
            f"""
            <div class="m3-metric-card">
              <div class="m3-metric-label-container">
                <span class="m3-metric-label">{label}</span>
                <span class="m3-metric-info-icon" data-tooltip="{metric_help.get(label, '')}">i</span>
              </div>
              <div class="m3-metric-value">{value_text}</div>
              <div class="m3-metric-progress-container">
                <div class="m3-metric-progress-bg">
                  <div class="m3-metric-progress-fill {status_color}" style="width:{progress:.2f}%;"></div>
                </div>
                <div class="m3-metric-footer">
                  <span class="m3-metric-delta {status_text}">{footer}</span>
                  <span class="m3-metric-threshold">{t_threshold}</span>
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

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown(f"#### {T[lang]['chart_top_features']}")
        top_features = top_features_frame(sources.external_report, limit=8)
        if top_features.empty:
            st.info(T[lang]["info_unavailable"])
        else:
            top_features_trans = top_features.rename(columns={
                "Feature": T[lang]["col_feature"],
                "Importance": T[lang]["col_importance"]
            })
            st.altair_chart(_bar_chart(top_features_trans, T[lang]["col_feature"], T[lang]["col_importance"]), width="stretch")
    with right:
        st.markdown(f"#### {T[lang]['chart_distribution']}")
        distribution = support_distribution_frame(sources.external_report, sources.manifest)
        if distribution.empty:
            st.info(T[lang]["info_unavailable"])
        else:
            distribution_trans = distribution.rename(columns={
                "Classe": T[lang]["col_class"],
                "Support": T[lang]["col_support"]
            })
            st.altair_chart(_bar_chart(distribution_trans, T[lang]["col_class"], T[lang]["col_support"]), width="stretch")


def render_analysis(sources, lang: str) -> None:
    st.markdown(f"<h2 class='ids-section-title'>{T[lang]['analysis_section_title']}</h2>", unsafe_allow_html=True)
    if not sources.external_report:
        st.info(T[lang]["info_unavailable"])
        return

    with st.expander(T[lang]["analysis_help_title"], expanded=False):
        st.markdown(T[lang]["analysis_help_content"])

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown(f"#### {T[lang]['analysis_confusion_matrix']}")
        confusion = confusion_matrix_frame(sources.external_report, sources.manifest)
        confusion_trans = confusion.reset_index(names=T[lang]["col_class"])
        _render_table(confusion_trans)
    with right:
        st.markdown(f"#### {T[lang]['analysis_class_distribution']}")
        distribution = support_distribution_frame(sources.external_report, sources.manifest)
        if not distribution.empty:
            distribution_trans = distribution.rename(columns={
                "Classe": T[lang]["col_class"],
                "Support": T[lang]["col_support"]
            })
            st.altair_chart(_bar_chart(distribution_trans, T[lang]["col_class"], T[lang]["col_support"]), width="stretch")

    metrics_col, features_col = st.columns([1, 1])
    with metrics_col:
        st.markdown(f"#### {T[lang]['analysis_metrics_class']}")
        classif = classification_frame(sources.external_report, sources.manifest)
        classif_trans = classif.rename(columns={
            "Classe": T[lang]["col_class"],
            "Precision": T[lang]["col_precision"],
            "Recall": T[lang]["col_recall"],
            "F1": T[lang]["col_f1"],
            "Support": T[lang]["col_support"]
        })
        _render_table(classif_trans)
    with features_col:
        st.markdown(f"#### {T[lang]['analysis_top_features']}")
        top_features = top_features_frame(sources.external_report)
        if not top_features.empty:
            top_features_trans = top_features.rename(columns={
                "Feature": T[lang]["col_feature"],
                "Importance": T[lang]["col_importance"]
            })
            st.altair_chart(_bar_chart(top_features_trans, T[lang]["col_feature"], T[lang]["col_importance"]), width="stretch")


def _get_presets(manifest, runtime_bundle=None, lang: str = "en") -> dict[str, dict[str, Any]]:
    preferred_dataset = Path("data/raw/KDDTest+.txt")
    sample = simulation_samples(manifest, preferred_path=preferred_dataset, max_rows=0)
    if sample.empty:
        sample = simulation_samples(manifest, max_rows=0)

    is_en = lang == "en"
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
            "name": "HTTP Web Traffic" if is_en else "Trafic Web HTTP",
            "desc": "Legitimate Web navigation on a healthy service." if is_en else "Navigation Web légitime sur un service sain.",
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
            "name": "Healthy SMTP Session" if is_en else "Session SMTP Saine",
            "desc": "Legitimate email sending without suspicious behavior." if is_en else "Envoi d'e-mail légitime sans comportement suspect.",
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
            "name": "TCP SYN Flood" if is_en else "TCP SYN Flood",
            "desc": "Service saturation through massive TCP connections." if is_en else "Saturation du service par ouverture TCP massive.",
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
            "name": "Port Scan" if is_en else "Scan de Ports",
            "desc": "Port scanning to map the network target." if is_en else "Balayage de ports pour cartographier la cible.",
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
            "name": "Ping of Death" if is_en else "Ping de la Mort",
            "desc": "Giant and malformed ICMP packets." if is_en else "Paquets ICMP géants et malformés.",
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
            "name": "SQL Injection" if is_en else "Injection SQL",
            "desc": "Attempt to extract database data via application query." if is_en else "Tentative d'extraction de données par requête applicative.",
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
            "name": "Privilege Escalation" if is_en else "Élévation de Privilèges",
            "desc": "Memory exploit aiming for an admin shell." if is_en else "Exploit mémoire visant un shell administrateur.",
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
            "name": "SSH Brute Force" if is_en else "Brute Force SSH",
            "desc": "Credential spraying on a remote access port." if is_en else "Spraying de credentials sur un accès distant.",
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
            "name": "Teardrop Fragment" if is_en else "Fragment Teardrop",
            "desc": "Overlapping IP fragments to destabilize the network stack." if is_en else "Fragments IP superposés pour déstabiliser la pile.",
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
            "name": "Reverse Shell Backdoor" if is_en else "Backdoor Reverse Shell",
            "desc": "Persistent inbound or outbound connection to an attacker." if is_en else "Connexion persistante entrante ou sortante vers un attaquant.",
        },
    }

    if not sample.empty:
        _hydrate_presets_from_validation(presets, sample, runtime_bundle)

    for key, preset in presets.items():
        preset["preset_key"] = key
    return presets


def _hydrate_presets_from_validation(
    presets: dict[str, dict[str, Any]],
    sample: pd.DataFrame,
    runtime_bundle=None,
) -> None:
    for scenario_key, label_candidates in SCENARIO_LABEL_CANDIDATES.items():
        preset = presets.get(scenario_key)
        if preset is None:
            continue

        row = _select_validation_row(sample, label_candidates, runtime_bundle)
        if row is None:
            continue

        for field in (*NSL_KDD_COLUMNS, "label", "category"):
            if field in row:
                preset[field] = row[field]
        preset["validation_label"] = row.get("label", preset.get("label", "unknown"))
        preset["validation_category"] = row.get("category", preset.get("category", "unknown"))


def _select_validation_row(sample: pd.DataFrame, label_candidates: tuple[str, ...], runtime_bundle=None):
    fallback = None
    for label in label_candidates:
        rows = sample[sample["label"] == label]
        if rows.empty:
            continue
        if fallback is None:
            fallback = rows.iloc[0].to_dict()

        if runtime_bundle is None:
            return fallback

        limited_rows = rows.head(400)
        payloads = [
            build_payload({field: row[field] for field in NSL_KDD_COLUMNS})
            for _, row in limited_rows.iterrows()
        ]
        predictions = predict_batch(runtime_bundle, payloads).predictions
        for index, prediction in enumerate(predictions):
            if prediction.label != "normal":
                return limited_rows.iloc[index].to_dict()

    return fallback


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
    lang: str = "en",
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
        .replace("__LANG__", str(lang))
    )
    components.html(html_content, height=330)


def render_tester(sources, lang: str) -> None:
    st.markdown(f"<h2 class='ids-section-title'>{T[lang]['tester_section_title']}</h2>", unsafe_allow_html=True)
    if not sources.runtime_available:
        st.info(T[lang]["tester_not_available"])
        return

    runtime_bundle = _load_runtime(str(sources.artifact_dir))
    presets = _get_presets(sources.manifest, runtime_bundle, lang)

    group_names = {
        "Flux legitimes": "Flux Légitimes" if lang == "fr" else "Legitimate Traffic",
        "Attaques reseau": "Attaques Réseau" if lang == "fr" else "Network Attacks",
        "Exploits applicatifs": "Exploits Applicatifs" if lang == "fr" else "Application Exploits",
    }

    scenario_groups = {
        "Flux legitimes": ["normal_http", "normal_smtp"],
        "Attaques reseau": ["ddos_syn", "teardrop", "nmap_scan", "ping_death"],
        "Exploits applicatifs": ["sql_injection", "ssh_bruteforce", "buffer_overflow", "backdoor"],
    }

    translated_groups = {group_names[k]: v for k, v in scenario_groups.items()}

    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        selected_group_trans = st.selectbox(T[lang]["tester_family"], list(translated_groups.keys()), key="tester-group")
        internal_group = [k for k, v in group_names.items() if v == selected_group_trans][0]
    with sel_col2:
        selected_key = st.selectbox(
            T[lang]["tester_scenario"],
            translated_groups[selected_group_trans],
            key="tester-scenario",
            format_func=lambda key: presets[key]["name"],
        )

    selected_preset = {**presets[selected_key], "preset_key": selected_key}

    col_left, col_right = st.columns([0.45, 0.55])
    with col_left:
        st.markdown(
            f"""
            <div class="ids-card ids-reveal ids-reveal-delay-1" style="margin-bottom: 1.5rem; height: calc(100% - 60px);">
              <div class="ids-scenario-kicker">{selected_group_trans}</div>
              <h3 class="ids-scenario-title">{selected_preset['name']}</h3>
              <p class="ids-scenario-desc">{selected_preset['desc']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(T[lang]["tester_start_simulation"], key="run-selected-scenario", type="primary", use_container_width=True):
            st.session_state["active_payload"] = selected_preset
            st.session_state["simulation_source"] = "preset"
            st.session_state["simulation_triggered"] = True
            st.session_state["last_run_key"] = selected_key
            st.rerun()
    with col_right:
        _render_scenario_summary(selected_preset, lang)

    # Reset simulation if the user changed the scenario without clicking the button
    if st.session_state.get("last_run_key") != selected_key and st.session_state.get("simulation_source") == "preset":
        st.session_state["simulation_triggered"] = False

    active_payload = st.session_state.get("active_payload")
    simulation_triggered = st.session_state.get("simulation_triggered", False)
    if not simulation_triggered or active_payload is None:
        _render_interactive_simulator(
            status="idle",
            label="N/A",
            score=0.0,
            threshold=0.5,
            category="N/A",
            scenario="idle",
            lang=lang,
        )
        st.info(T[lang]["tester_waiting_info"])
    else:
        _render_simulation_result(sources, active_payload, lang)

    with st.expander(T[lang]["tester_manual_title"], expanded=False):
        values: dict[str, Any] = {}
        default_source = active_payload if active_payload is not None else DEFAULT_PAYLOAD
        with st.form("prediction-form"):
            cat_cols = st.columns(3)
            protocol_value = str(default_source.get("protocol_type", "tcp"))
            service_value = str(default_source.get("service", "http"))
            flag_value = str(default_source.get("flag", "SF"))
            values["protocol_type"] = cat_cols[0].selectbox(
                T[lang]["tester_protocol"],
                PROTOCOL_OPTIONS,
                index=PROTOCOL_OPTIONS.index(protocol_value) if protocol_value in PROTOCOL_OPTIONS else 0,
            )
            values["service"] = cat_cols[1].selectbox(
                T[lang]["tester_service"],
                SERVICE_OPTIONS,
                index=SERVICE_OPTIONS.index(service_value) if service_value in SERVICE_OPTIONS else 0,
            )
            values["flag"] = cat_cols[2].selectbox(
                T[lang]["tester_flag"],
                FLAG_OPTIONS,
                index=FLAG_OPTIONS.index(flag_value) if flag_value in FLAG_OPTIONS else 0,
            )

            st.markdown(f"#### {T[lang]['tester_main_signals']}")
            for row in _chunked(IMPORTANT_NUMERIC_FIELDS, 4):
                columns = st.columns(len(row))
                for column, field in zip(columns, row):
                    values[field] = column.text_input(
                        field,
                        value=_format_numeric_value(default_source.get(field, DEFAULT_PAYLOAD[field])),
                        key=f"main-{field}",
                    )

            with st.expander(T[lang]["tester_advanced_fields"]):
                for row in _chunked(ADVANCED_FIELDS, 4):
                    columns = st.columns(len(row))
                    for column, field in zip(columns, row):
                        values[field] = column.text_input(
                            field,
                            value=_format_numeric_value(default_source.get(field, DEFAULT_PAYLOAD[field])),
                            key=f"advanced-{field}",
                        )

            submitted = st.form_submit_button(T[lang]["tester_analyze_button"], disabled=not sources.runtime_available)

        if submitted:
            try:
                payload = build_payload(_coerce_numeric_values(values))
                st.session_state["active_payload"] = payload
                st.session_state["simulation_source"] = "manual"
                st.session_state["simulation_triggered"] = True
                st.session_state["last_run_key"] = None
                st.rerun()
            except Exception as exc:
                st.error(T[lang]["tester_input_error"].format(exc=exc))


def _render_scenario_summary(preset: dict[str, Any], lang: str) -> None:
    t_sig = "Signal" if lang == "fr" else "Signal"
    t_val = "Valeur" if lang == "fr" else "Value"

    summary = pd.DataFrame(
        [
            {t_sig: "protocol_type", t_val: str(preset.get("protocol_type", "tcp"))},
            {t_sig: "service", t_val: str(preset.get("service", "http"))},
            {t_sig: "flag", t_val: str(preset.get("flag", "SF"))},
            {t_sig: "label_validation", t_val: str(preset.get("validation_label", preset.get("label", "unknown")))},
            {t_sig: "src_bytes", t_val: str(preset.get("src_bytes", 0))},
            {t_sig: "dst_bytes", t_val: str(preset.get("dst_bytes", 0))},
            {t_sig: "count", t_val: str(preset.get("count", 0))},
            {t_sig: "srv_count", t_val: str(preset.get("srv_count", 0))},
        ]
    )
    st.markdown("<div class='ids-card ids-reveal ids-reveal-delay-2' style='height: 100%;'>", unsafe_allow_html=True)
    _render_table(summary)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_simulation_result(sources, payload: dict[str, Any], lang: str) -> None:
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
        lang=lang,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    badge_html = '<span class="ids-pill">NORMAL</span>' if status == "normal" else ""
    decision = T[lang]["decision_allowed"] if status == "normal" else T[lang]["decision_blocked"]

    details = pd.DataFrame(
        [
            {T[lang]["col_step"]: T[lang]["step_flow_type"], T[lang]["col_value"]: str(payload.get("name", T[lang]["tester_manual_flow"]))},
            {T[lang]["col_step"]: T[lang]["step_validation_label"], T[lang]["col_value"]: str(payload.get("validation_label", payload.get("label", "unknown")))},
            {T[lang]["col_step"]: T[lang]["step_predicted_category"], T[lang]["col_value"]: result.category},
            {T[lang]["col_step"]: T[lang]["step_ml_label"], T[lang]["col_value"]: result.label},
            {T[lang]["col_step"]: T[lang]["step_method"], T[lang]["col_value"]: detection_method},
            {T[lang]["col_step"]: T[lang]["step_score"], T[lang]["col_value"]: f"{result.score:.3f}"},
            {T[lang]["col_step"]: T[lang]["step_threshold"], T[lang]["col_value"]: f"{result.threshold:.3f}"},
            {T[lang]["col_step"]: T[lang]["tester_final_action"], T[lang]["col_value"]: "ALLOW" if status == "normal" else "BLOCK"},
        ]
    )
    st.markdown(
        f"""
        <div class="ids-card ids-reveal ids-result-delayed" style="margin-top: 1rem;">
          <div class="ids-result-head">
            <div>
              <h3 class="ids-result-title">{decision}</h3>
              <p class="ids-result-subtitle">{T[lang]["decision_subtitle"]}</p>
            </div>
            {badge_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_table(details)


def render_runtime(sources, lang: str) -> None:
    st.markdown(f"<h2 class='ids-section-title'>{T[lang]['runtime_section_title']}</h2>", unsafe_allow_html=True)
    if not sources.manifest:
        st.info(T[lang]["runtime_no_manifest"])
    else:
        summary = runtime_summary(sources.manifest)
        runtime_rows = [
            {T[lang]["col_field"]: "Modèle" if lang == "fr" else "Model", T[lang]["col_value"]: str(summary.get("model_name"))},
            {T[lang]["col_field"]: "Profil" if lang == "fr" else "Profile", T[lang]["col_value"]: str(summary.get("profile_name"))},
            {T[lang]["col_field"]: "Seuil" if lang == "fr" else "Threshold", T[lang]["col_value"]: str(summary.get("threshold"))},
            {T[lang]["col_field"]: "Features", T[lang]["col_value"]: str(summary.get("feature_count"))},
            {T[lang]["col_field"]: "Dataset", T[lang]["col_value"]: str(summary.get("dataset_path"))},
        ]
        _render_table(pd.DataFrame(runtime_rows))
        mapping_rows = [{T[lang]["col_key"]: key, "Index": value} for key, value in summary.get("label_mapping", {}).items()]
        hash_rows = [{T[lang]["col_file"]: key, T[lang]["col_hash"]: value} for key, value in summary.get("artifact_hashes", {}).items()]
        left, right = st.columns(2)
        with left:
            st.markdown(f"#### {T[lang]['runtime_label_mapping']}")
            if mapping_rows:
                _render_table(pd.DataFrame(mapping_rows))
            else:
                st.info(T[lang]["info_unavailable"])
        with right:
            st.markdown(f"#### {T[lang]['runtime_artifact_integrity']}")
            if hash_rows:
                _render_table(pd.DataFrame(hash_rows))
            else:
                st.info(T[lang]["info_unavailable"])

    st.warning(T[lang]["runtime_warning"])
    if sources.runtime_available:
        bundle = _load_runtime(str(sources.artifact_dir))
        st.markdown(f"#### {T[lang]['runtime_loaded_title']}")
        runtime_details = describe_runtime(bundle)
        runtime_frame = pd.DataFrame(
            [{T[lang]["col_key"]: key, T[lang]["col_value"]: str(value)} for key, value in runtime_details.items() if key != "feature_columns"]
        )
        _render_table(runtime_frame)
        with st.expander(T[lang]["runtime_feature_columns"]):
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
      display: flex;
      overflow-x: auto;
      gap: 16px;
      width: 100%;
      box-sizing: border-box;
      padding-bottom: 8px;
      scrollbar-width: thin;
      scrollbar-color: #c4c7c5 transparent;
    }
    .m3-metric-grid::-webkit-scrollbar {
      height: 4px;
    }
    .m3-metric-grid::-webkit-scrollbar-thumb {
      background: #c4c7c5;
      border-radius: 99px;
    }
    .m3-metric-card {
      background: #ffffff;
      border: 1px solid #c4c7c5;
      border-radius: 16px;
      padding: 1.1rem;
      min-height: 128px;
      flex: 1 1 0px;
      min-width: 180px;
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
      position: relative;
      cursor: help;
    }
    .m3-metric-info-icon::after {
      content: attr(data-tooltip);
      position: absolute;
      top: 130%;
      right: 0;
      background-color: #1f1f1f;
      color: #ffffff;
      padding: 8px 12px;
      border-radius: 8px;
      font-size: 0.72rem;
      font-weight: 500;
      white-space: normal;
      width: 180px;
      box-shadow: 0 4px 6px rgba(0,0,0,0.15);
      opacity: 0;
      visibility: hidden;
      transition: opacity 0.2s, visibility 0.2s;
      z-index: 99;
      font-family: Inter, Arial, sans-serif;
      text-transform: none;
      line-height: 1.3;
    }
    .m3-metric-info-icon:hover::after {
      opacity: 1;
      visibility: visible;
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
    """


if __name__ == "__main__":
    main(sys.argv[1:])
