from __future__ import annotations

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;700&family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
  --m3-primary: #0b57d0;
  --m3-primary-hover: #084bb5;
  --m3-primary-container: #d3e3fd;
  --m3-on-primary-container: #041e49;
  --m3-surface: #ffffff;
  --m3-surface-container: #f0f4f9;
  --m3-surface-variant: #e1e2e9;
  --m3-outline: #c4c7c5;
  --m3-bg: #f8f9fa;
  --m3-text: #1f1f1f;
  --m3-text-muted: #5e5e5e;
  --m3-success: #146c2e;
  --m3-success-container: #e8f5e9;
  --m3-error: #b3261e;
  --m3-error-container: #f9dedc;
  --m3-warning: #b06000;
  --m3-warning-container: #fff3e0;
}

/* Global modifications for Material 3 Light Theme */
html, body, [data-testid="stAppViewContainer"] {
  font-family: 'Inter', sans-serif;
  background-color: var(--m3-bg) !important;
  color: var(--m3-text) !important;
  overflow-x: hidden !important;
  width: 100% !important;
}

.stApp {
  background: var(--m3-bg) !important;
  color-scheme: light !important;
  overflow-x: hidden !important;
}

header[data-testid="stHeader"],
[data-testid="stHeader"] {
  display: none !important;
}

#MainMenu {
  visibility: hidden;
}

.block-container {
  max-width: 1500px;
  padding: 2rem 2.5rem 3rem !important;
  background: transparent;
}

/* Hide default sidebar completely */
[data-testid="stSidebar"] {
  display: none !important;
}
[data-testid="collapsedSidebarButton"] {
  display: none !important;
}

/* Google Store Header Custom Styles */
.google-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  height: 40px;
}

.google-brand-text {
  font-weight: 700;
  font-size: 1.15rem;
  color: var(--m3-text);
  letter-spacing: -0.02em;
}

.google-header-right {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 20px;
  height: 40px;
}

.google-header-right svg {
  transition: opacity 0.2s;
}
.google-header-right svg:hover {
  opacity: 0.7;
}

.google-profile-img {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  border: 1px solid var(--m3-outline);
  object-fit: cover;
}

/* Headers and text styling */
h1, h2, h3, h4, h5, h6 {
  font-family: 'Inter', sans-serif;
  letter-spacing: -0.015em;
  color: var(--m3-text) !important;
}

[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
  color: var(--m3-text) !important;
}

[data-testid="stMarkdownContainer"] h1 {
  font-weight: 800 !important;
  font-size: 2.2rem !important;
}

[data-testid="stMarkdownContainer"] h2 {
  font-weight: 700 !important;
  font-size: 1.6rem !important;
  border-bottom: 0px !important;
}

.ids-topline {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 14px;
  border-radius: 999px;
  background-color: var(--m3-primary-container) !important;
  color: var(--m3-on-primary-container) !important;
  font-size: 0.80rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em;
  margin-bottom: 0.8rem;
  border: 1px solid rgba(11, 87, 208, 0.15);
}

.ids-page-title {
  color: var(--m3-text);
  font-size: 2rem;
  line-height: 1.2;
  margin: 0;
  font-weight: 800;
}

.ids-subtitle {
  color: var(--m3-text-muted);
  margin-top: 0.4rem;
  margin-bottom: 1.8rem;
  font-size: 0.92rem;
}

/* M3 Elevated & Outlined Cards */
.ids-card {
  background: var(--m3-surface);
  border: 1px solid var(--m3-outline);
  border-radius: 16px;
  padding: 1.3rem 1.6rem;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
  transition: all 0.2s cubic-bezier(0.2, 0, 0, 1);
}

.ids-card:hover {
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
  border-color: var(--m3-primary);
}

.ids-kpi-label {
  color: var(--m3-text-muted);
  font-size: 0.82rem;
  font-weight: 600;
  margin: 0;
}

.ids-kpi-value {
  color: var(--m3-text);
  font-size: 1.9rem;
  font-weight: 800;
  margin: 0.2rem 0 0;
}

/* M3 Tonal & Outlined Pills */
.ids-pill {
  display: inline-flex;
  align-items: center;
  min-height: 28px;
  padding: 0.25rem 0.75rem;
  border-radius: 8px;
  background: var(--m3-primary-container);
  color: var(--m3-on-primary-container);
  font-weight: 600;
  font-size: 0.75rem;
  border: 1px solid transparent;
  letter-spacing: 0.02em;
}

.ids-pill-danger {
  background: var(--m3-error-container) !important;
  color: var(--m3-error) !important;
  border-color: transparent !important;
}

.ids-section-title {
  margin-top: 1rem;
  margin-bottom: 0.8rem;
  color: var(--m3-text);
}

.ids-muted {
  color: var(--m3-text-muted);
}

/* Google Store Floating Navigation Pill Container - Outlined M3 card styling */
[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:first-of-type {
  background-color: #ffffff !important;
  border: 1px solid var(--m3-outline) !important;
  border-radius: 20px !important;
  padding: 12px 28px !important;
  margin-bottom: 2rem !important;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
  transition: all 0.25s cubic-bezier(0.2, 0, 0, 1) !important;
}

[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:first-of-type:hover {
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06) !important;
  border-color: var(--m3-primary) !important;
}

/* Top Navigation Tabs Styling */
[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stButton"] button {
  background-color: transparent !important;
  color: var(--m3-text-muted) !important;
  border: none !important;
  font-weight: 500 !important;
  font-size: 0.92rem !important;
  min-height: 38px !important;
  padding: 0 18px !important;
  border-radius: 999px !important; /* capsule shape */
  transition: all 0.2s cubic-bezier(0.2, 0, 0, 1) !important;
  box-shadow: none !important;
}

[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stButton"] button:hover {
  background-color: rgba(31, 31, 31, 0.05) !important;
  color: var(--m3-text) !important;
}

[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:first-of-type div[data-testid="stBaseButton-primary"] button {
  background-color: var(--m3-surface-container) !important; /* soft light gray pill */
  color: var(--m3-primary) !important;
  font-weight: 600 !important;
}

/* Sidebar Status Pill Box */
.m3-sidebar-status {
  margin: 2rem 0.8rem 1rem;
}

.m3-status-pill {
  display: inline-flex;
  align-items: center;
  padding: 6px 14px;
  border-radius: 999px;
  font-size: 0.74rem;
  font-weight: 600;
}

.m3-status-pill.success {
  background-color: var(--m3-success-container);
  color: var(--m3-success);
}

.m3-status-pill.danger {
  background-color: var(--m3-error-container);
  color: var(--m3-error);
}

/* Keep columns perfectly horizontal in the topbar even on small screens */
[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:first-of-type {
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: nowrap !important;
  align-items: center !important;
  justify-content: space-between !important;
  gap: 12px !important;
}

[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:first-of-type > div {
  width: auto !important;
  min-width: 0 !important;
  flex-grow: 1 !important;
}

@media (max-width: 768px) {
  /* Hide G brand title on very small screens to save space */
  .google-brand-text {
    display: none !important;
  }
  .google-header-right {
    gap: 10px !important;
  }
  .ids-card {
    height: auto !important;
  }
}

@media (max-width: 600px) {
  .block-container {
    padding: 1rem 1rem 2rem !important;
  }
  .ids-page-title {
    font-size: 1.5rem !important;
  }
  .ids-subtitle {
    font-size: 0.82rem !important;
    margin-bottom: 1.2rem !important;
  }
  .ids-card {
    padding: 1rem 1.1rem !important;
  }
  [data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:first-of-type {
    overflow-x: auto !important;
    white-space: nowrap !important;
    justify-content: flex-start !important;
    padding: 8px 16px !important;
    margin-bottom: 1.2rem !important;
    -webkit-overflow-scrolling: touch !important;
    scrollbar-width: none !important; /* Firefox */
  }
  [data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:first-of-type::-webkit-scrollbar {
    display: none !important; /* Safari and Chrome */
  }
  [data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:first-of-type > div {
    flex-shrink: 0 !important;
    flex-grow: 0 !important;
    width: auto !important;
    min-width: fit-content !important;
  }
  [data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stButton"] button {
    white-space: nowrap !important;
  }
}


/* Tech paths box overrides - clean gray background instead of black */
.tech-paths {
  font-size: 0.74rem;
  color: var(--m3-text-muted);
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 8px;
  background: var(--m3-surface-container) !important;
  border-radius: 8px;
}

.tech-paths code {
  background-color: rgba(0, 0, 0, 0.05);
  padding: 2px 5px;
  border-radius: 4px;
  font-family: 'Fira Code', monospace;
  color: var(--m3-text) !important;
}

/* Standard Buttons styling - M3 Filled style */
/* Secondary Buttons - M3 Outlined / Tonal style */
div[data-testid="stBaseButton-secondary"] button,
.stButton button {
  background: var(--m3-surface) !important;
  color: var(--m3-primary) !important;
  border: 1px solid var(--m3-outline) !important;
  border-radius: 999px !important; /* M3 Pill shape */
  min-height: 38px;
  padding: 0.3rem 1.4rem;
  font-weight: 600;
  box-shadow: none !important;
  transition: all 0.2s cubic-bezier(0.2, 0, 0, 1);
}

div[data-testid="stBaseButton-secondary"] button:hover,
.stButton button:hover {
  background-color: rgba(11, 87, 208, 0.04) !important;
  border-color: var(--m3-primary) !important;
  color: var(--m3-primary-hover) !important;
}

div[data-testid="stBaseButton-secondary"] button:active,
.stButton button:active {
  transform: scale(0.98);
}

/* Primary / Form Submit Buttons - M3 Filled style */
div[data-testid="stBaseButton-primary"] button,
div[data-testid="stFormSubmitButton"] button {
  background: var(--m3-primary) !important;
  color: #ffffff !important;
  border-radius: 999px !important;
  border: none !important;
  min-height: 40px !important;
  padding: 0.4rem 1.6rem !important;
  font-weight: 600 !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
  transition: background-color 0.15s ease !important;
}

div[data-testid="stBaseButton-primary"] button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
  background-color: var(--m3-primary-hover) !important;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15) !important;
}

div[data-testid="stBaseButton-primary"] button:active,
div[data-testid="stFormSubmitButton"] button:active {
  transform: scale(0.98);
}

/* Custom inputs styling */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input,
div[data-baseweb="input"],
div[data-baseweb="input"] input,
div[data-baseweb="select"],
div[data-baseweb="select"] > div {
  background: var(--m3-surface) !important;
  color: var(--m3-text) !important;
  border: 1px solid var(--m3-outline) !important;
  border-radius: 8px;
}

div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"]:focus-within {
  border-color: var(--m3-primary) !important;
  box-shadow: 0 0 0 2px rgba(11, 87, 208, 0.2) !important;
}

.stSelectbox label,
.stNumberInput label,
.stTextInput label {
  color: var(--m3-text-muted);
  font-weight: 600;
  font-size: 0.82rem;
}

/* Metric M3 Card style */
div[data-testid="stMetric"] {
  background: var(--m3-surface) !important;
  border: 1px solid var(--m3-outline) !important;
  border-radius: 12px !important;
  padding: 0.8rem 1rem !important;
  box-shadow: none !important;
}

div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
  color: var(--m3-text-muted) !important;
  font-weight: 600 !important;
  font-size: 0.8rem !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: var(--m3-text) !important;
  font-weight: 800 !important;
  font-size: 1.7rem !important;
}

/* Streamlit Dataframe custom clear theme */
div[data-testid="stDataFrame"] {
  border: 1px solid var(--m3-outline) !important;
  border-radius: 12px !important;
  overflow: hidden;
  background: var(--m3-surface) !important;
}

div[data-testid="stDataFrame"] * {
  color: var(--m3-text) !important;
}

div[data-testid="stDataFrame"] div[class*="gdg-"],
div[data-testid="stDataFrame"] div[class*="gdg-"] * {
  background: var(--m3-surface) !important;
  color: var(--m3-text) !important;
}

div[data-testid="stDataFrame"] div[role="columnheader"] {
  background: var(--m3-surface-container) !important;
  color: var(--m3-text) !important;
}

[data-testid="stSidebarNav"] {
  display: none;
}

/* Custom Outlined tables */
.ids-table-wrap {
  width: 100%;
  overflow-x: auto;
  margin: 0.6rem 0 1.5rem;
  background: var(--m3-surface);
  border: 1px solid var(--m3-outline);
  border-radius: 12px;
}

.ids-table {
  width: 100%;
  border-collapse: collapse;
  background: transparent;
  color: var(--m3-text);
  font-size: 0.85rem;
}

.ids-table thead th {
  background: var(--m3-surface-container);
  color: var(--m3-text);
  font-weight: 700;
  text-align: left;
  padding: 0.8rem 1rem;
  border-bottom: 1px solid var(--m3-outline);
  font-size: 0.78rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.ids-table tbody td {
  padding: 0.75rem 1rem;
  border-bottom: 1px solid #e1e2e9;
  vertical-align: middle;
}

.ids-table tbody tr:last-child td {
  border-bottom: 0;
}

.ids-table tbody tr:hover td {
  background: var(--m3-surface-container);
}

.ids-scenario-kicker {
  color: var(--m3-primary);
  font-size: 0.76rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 0.45rem;
}

.ids-scenario-title {
  margin: 0;
  font-size: 1.15rem;
  font-weight: 700;
}

.ids-scenario-desc {
  margin: 0.45rem 0 0;
  color: var(--m3-text-muted);
  font-size: 0.9rem;
  line-height: 1.5;
}

.ids-result-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
}

.ids-result-title {
  margin: 0;
  font-size: 1.05rem;
  font-weight: 700;
}

.ids-result-subtitle {
  margin: 0.35rem 0 0;
  color: var(--m3-text-muted) !important;
  font-size: 0.88rem;
}

.ids-reveal {
  animation: ids-fade-up 360ms cubic-bezier(0.2, 0, 0, 1) both;
}

.ids-reveal-delay-1 {
  animation-delay: 60ms;
}

.ids-reveal-delay-2 {
  animation-delay: 140ms;
}

.ids-result-delayed {
  animation-delay: 2600ms;
}

@keyframes ids-fade-up {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Streamlit Expander clear M3 style */
[data-testid="stExpander"] {
  background: var(--m3-surface) !important;
  border: 1px solid var(--m3-outline) !important;
  border-radius: 12px !important;
  box-shadow: none !important;
  overflow: hidden;
}

[data-testid="stExpander"] details summary {
  background: var(--m3-surface) !important;
  color: var(--m3-text) !important;
  font-weight: 600 !important;
}

[data-testid="stExpander"] details summary:hover {
  background: var(--m3-surface-container) !important;
  color: var(--m3-primary) !important;
}

[data-testid="stExpander"] details[open] summary {
  background: var(--m3-surface-container) !important;
  border-bottom: 1px solid var(--m3-outline) !important;
}

[data-testid="stExpander"] details summary:active,
[data-testid="stExpander"] details summary:focus,
[data-testid="stExpander"] details summary * {
  background: transparent !important;
}

/* Hide Streamlit Deploy Button completely */
div[data-testid="stDeployButton"],
header [data-testid="stDeployButton"] {
  display: none !important;
}

/* Custom M3 Metric Cards Grid & Components */
.m3-metric-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 16px;
  margin-bottom: 1.5rem;
  width: 100%;
}

@media (max-width: 992px) {
  .m3-metric-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (max-width: 768px) {
  .m3-metric-grid {
    grid-template-columns: 1fr;
  }
}

.m3-metric-card {
  background: var(--m3-surface);
  border: 1px solid var(--m3-outline);
  border-radius: 16px;
  padding: 1.2rem;
  box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-height: 135px;
  transition: all 0.2s cubic-bezier(0.2, 0, 0, 1);
  position: relative;
}

.m3-metric-card:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  border-color: var(--m3-primary);
}

.m3-metric-label-container {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.m3-metric-label {
  color: var(--m3-text-muted) !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  margin: 0 !important;
}

.m3-metric-info-icon {
  color: var(--m3-text-muted);
  font-size: 0.74rem;
  cursor: help;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  border: 1px solid var(--m3-outline);
  background: transparent;
  transition: all 0.2s;
  font-weight: bold;
}

.m3-metric-card:hover .m3-metric-info-icon {
  color: var(--m3-primary);
  border-color: var(--m3-primary);
  background: rgba(11, 87, 208, 0.04);
}

.m3-metric-value {
  color: var(--m3-text) !important;
  font-size: 2.1rem !important;
  font-weight: 800 !important;
  margin: 0.2rem 0 !important;
  line-height: 1 !important;
  font-family: 'Inter', sans-serif !important;
  letter-spacing: -0.02em !important;
}

.m3-metric-progress-container {
  margin-top: auto;
  padding-top: 4px;
}

.m3-metric-progress-bg {
  width: 100%;
  height: 6px;
  background-color: var(--m3-surface-container);
  border-radius: 99px;
  overflow: hidden;
  margin-bottom: 8px;
}

.m3-metric-progress-fill {
  height: 100%;
  border-radius: 99px;
  transition: width 0.6s cubic-bezier(0.2, 0, 0, 1);
}

.m3-color-success {
  background-color: #146c2e !important;
}

.m3-color-error {
  background-color: #b3261e !important;
}

.m3-color-gray {
  background-color: #c4c7c5 !important;
}

.m3-metric-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.74rem;
  font-weight: 600;
}

.m3-text-success {
  color: #146c2e !important;
}

.m3-text-error {
  color: #b3261e !important;
}

.m3-text-gray {
  color: #5e5e5e !important;
}

.m3-metric-threshold {
  color: var(--m3-text-muted) !important;
  font-weight: 500 !important;
}
</style>
"""
