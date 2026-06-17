"""SOC dashboard global CSS theme."""

from __future__ import annotations

import streamlit as st


def inject_soc_theme() -> None:
    """Inject cyber-security SOC styling (dark theme, glow cards, compact sidebar)."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .block-container {
            padding-top: 1rem;
            padding-bottom: 0.5rem;
            max-width: 1400px;
        }

        /* ── Hero (önceki başlık) ── */
        .hero-header {
            background: linear-gradient(135deg, #6C63FF 0%, #3F3D9E 50%, #1A1D29 100%);
            padding: 1.5rem 1.25rem;
            border-radius: 14px;
            margin-bottom: 0.75rem;
            text-align: center;
            box-shadow: 0 8px 32px rgba(108,99,255,0.2);
            border: 1px solid rgba(108,99,255,0.2);
        }
        .hero-header h1 {
            color: #FFFFFF;
            font-size: 1.75rem;
            font-weight: 800;
            margin: 0;
        }
        .hero-header p {
            color: rgba(255,255,255,0.8);
            font-size: 0.9rem;
            margin-top: 0.35rem;
        }

        /* ── SOC header ── */
        .soc-header {
            background: linear-gradient(90deg, #0a0e17 0%, #121a2e 40%, #0d1525 100%);
            border: 1px solid rgba(0, 255, 170, 0.2);
            border-radius: 10px;
            padding: 0.75rem 1.25rem;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 0 24px rgba(0, 255, 170, 0.06);
        }
        .soc-header h1 {
            color: #e8f4ff;
            font-size: 1.35rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: 0.5px;
        }
        .soc-header .sub {
            color: #6b8cae;
            font-size: 0.78rem;
            margin-top: 2px;
        }
        .soc-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            padding: 0.25rem 0.65rem;
            border-radius: 4px;
            font-weight: 600;
            letter-spacing: 1px;
        }
        .soc-badge-live {
            background: rgba(0, 255, 136, 0.12);
            color: #00ff88;
            border: 1px solid rgba(0, 255, 136, 0.35);
            animation: pulse-live 2s ease-in-out infinite;
        }
        @keyframes pulse-live {
            0%, 100% { box-shadow: 0 0 6px rgba(0,255,136,0.2); }
            50% { box-shadow: 0 0 14px rgba(0,255,136,0.45); }
        }

        /* ── KPI strip ── */
        .kpi-card {
            background: rgba(15, 22, 38, 0.9);
            border: 1px solid rgba(100, 140, 200, 0.15);
            border-radius: 8px;
            padding: 0.55rem 0.75rem;
            text-align: center;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        .kpi-card:hover {
            border-color: rgba(0, 200, 255, 0.35);
            box-shadow: 0 4px 16px rgba(0, 120, 255, 0.1);
        }
        .kpi-value {
            font-size: 1.35rem;
            font-weight: 700;
            color: #4dc9ff;
            font-family: 'JetBrains Mono', monospace;
        }
        .kpi-label {
            font-size: 0.68rem;
            color: #6b8cae;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-top: 2px;
        }

        /* ── Prediction glow cards ── */
        .pred-card-attack {
            background: linear-gradient(135deg, rgba(255,40,60,0.22), rgba(80,0,20,0.35));
            border: 1px solid rgba(255, 60, 80, 0.55);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            text-align: center;
            box-shadow: 0 0 28px rgba(255, 50, 70, 0.35), inset 0 0 40px rgba(255,0,0,0.05);
            animation: glow-red 2.5s ease-in-out infinite;
        }
        .pred-card-attack h2 {
            color: #ff4466;
            font-size: 1.45rem;
            font-weight: 800;
            margin: 0;
            text-shadow: 0 0 20px rgba(255,60,80,0.6);
        }
        .pred-card-normal {
            background: linear-gradient(135deg, rgba(0,220,120,0.18), rgba(0,60,40,0.25));
            border: 1px solid rgba(0, 255, 140, 0.45);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            text-align: center;
            box-shadow: 0 0 24px rgba(0, 255, 140, 0.2), inset 0 0 30px rgba(0,255,100,0.04);
        }
        .pred-card-normal h2 {
            color: #00e87a;
            font-size: 1.45rem;
            font-weight: 800;
            margin: 0;
            text-shadow: 0 0 16px rgba(0,255,120,0.4);
        }
        @keyframes glow-red {
            0%, 100% { box-shadow: 0 0 20px rgba(255,50,70,0.25); }
            50% { box-shadow: 0 0 36px rgba(255,50,70,0.5); }
        }
        .pred-meta {
            color: #8fa8c8;
            font-size: 0.85rem;
            margin-top: 0.5rem;
        }
        .threat-pill {
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.2rem 0.7rem;
            border-radius: 4px;
            letter-spacing: 1.5px;
            margin-top: 0.5rem;
        }

        /* ── Feature micro-cards ── */
        .feat-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(20, 30, 50, 0.8);
            border: 1px solid rgba(80, 120, 180, 0.2);
            border-radius: 6px;
            padding: 0.3rem 0.65rem;
            margin: 0.2rem 0.25rem 0.2rem 0;
            font-size: 0.8rem;
            transition: background 0.15s, border-color 0.15s;
        }
        .feat-chip:hover {
            background: rgba(30, 50, 80, 0.9);
            border-color: rgba(0, 200, 255, 0.35);
        }
        .feat-chip .fname { color: #9eb8d8; }
        .feat-chip .fval {
            color: #4dc9ff;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
        }

        /* ── Section panel ── */
        .panel-title {
            color: #5a8ab0;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }

        /* ── Match badges ── */
        .match-ok {
            background: rgba(0,255,136,0.1);
            border: 1px solid rgba(0,255,136,0.35);
            color: #00ff88;
            padding: 0.5rem;
            border-radius: 6px;
            text-align: center;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .match-bad {
            background: rgba(255,60,80,0.1);
            border: 1px solid rgba(255,60,80,0.35);
            color: #ff5577;
            padding: 0.5rem;
            border-radius: 6px;
            text-align: center;
            font-weight: 600;
            font-size: 0.85rem;
        }

        .sidebar-header {
            text-align: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 0.75rem;
        }
        .sidebar-header h2 {
            color: #6C63FF;
            font-size: 1.2rem;
            font-weight: 700;
            margin: 0;
        }
        .info-box {
            background: rgba(108,99,255,0.08);
            border: 1px solid rgba(108,99,255,0.2);
            border-radius: 8px;
            padding: 0.75rem;
            font-size: 0.85rem;
        }
        .glass-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
        }
        .stat-card {
            background: linear-gradient(135deg, rgba(108,99,255,0.1), rgba(108,99,255,0.02));
            border: 1px solid rgba(108,99,255,0.15);
            border-radius: 10px;
            padding: 0.75rem;
            text-align: center;
        }
        .stat-card .stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #6C63FF;
        }
        .stat-card .stat-label {
            font-size: 0.72rem;
            color: rgba(224,224,224,0.65);
            text-transform: uppercase;
        }

        /* ── Sidebar compact ── */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #080c14 0%, #0f1520 100%);
            border-right: 1px solid rgba(0, 200, 255, 0.12);
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 0.75rem;
        }
        section[data-testid="stSidebar"] [data-testid="stMetric"] {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 6px;
            padding: 6px 10px;
        }
        section[data-testid="stSidebar"] [data-testid="stMetric"] label {
            font-size: 0.72rem !important;
        }
        section[data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1rem !important;
        }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: rgba(10,16,28,0.8);
            border-radius: 8px;
            padding: 3px;
            border: 1px solid rgba(80,120,180,0.12);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px;
            padding: 8px 14px;
            font-size: 0.85rem;
            font-weight: 500;
        }

        /* ── Buttons ── */
        .stButton > button {
            background: linear-gradient(135deg, #0d4a6e, #0a3050);
            color: #c8e8ff;
            border: 1px solid rgba(0, 180, 255, 0.35);
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s;
        }
        .stButton > button:hover {
            border-color: rgba(0, 220, 255, 0.6);
            box-shadow: 0 0 16px rgba(0, 180, 255, 0.25);
        }

        /* ── SHAP bar colors ── */
        div[data-testid="stHorizontalBlock"] { gap: 0.5rem; }

        hr { border-color: rgba(80,120,180,0.12) !important; margin: 0.5rem 0 !important; }

        /* ── SOC status strip ── */
        .soc-status-strip {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(8, 14, 24, 0.95);
            border: 1px solid rgba(0, 200, 255, 0.2);
            border-radius: 8px;
            padding: 0.45rem 1rem;
            margin-bottom: 0.65rem;
            font-size: 0.78rem;
        }
        .soc-status-left, .soc-status-right {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .soc-pulse {
            width: 8px; height: 8px;
            border-radius: 50%;
            display: inline-block;
            animation: pulse-live 2s infinite;
        }
        .soc-status-meta { color: #6b8cae; }
        .soc-status-sep { color: #3a5070; }

        /* ── Class badges ── */
        .class-badge {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.3px;
        }
        .meta-chip {
            display: inline-block;
            padding: 0.2rem 0.5rem;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 4px;
            font-size: 0.75rem;
            color: #9eb8d8;
        }
        .meta-line { color: #6b8cae; font-size: 0.8rem; margin-top: 0.35rem; }

        .section-card {
            background: rgba(12, 18, 30, 0.85);
            border: 1px solid rgba(80, 120, 180, 0.15);
            border-radius: 10px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.65rem;
        }
        .section-card-title {
            color: #8fa8c8;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }

        .sidebar-sub {
            color: #6b8cae !important;
            font-size: 0.75rem !important;
            margin: 0.15rem 0 0 0 !important;
        }
        .sidebar-online {
            text-align: center;
            color: #00ff88;
            font-size: 0.78rem;
            font-weight: 600;
            padding: 0.35rem;
            background: rgba(0,255,136,0.08);
            border-radius: 6px;
            margin-bottom: 0.5rem;
            border: 1px solid rgba(0,255,136,0.25);
        }

        /* ── Probability bars (HTML) ── */
        .prob-row {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin: 0.35rem 0;
            font-size: 0.8rem;
        }
        .prob-label {
            width: 52px;
            color: #9eb8d8;
            font-weight: 600;
            flex-shrink: 0;
        }
        .prob-track {
            flex: 1;
            height: 10px;
            background: rgba(255,255,255,0.06);
            border-radius: 5px;
            overflow: hidden;
        }
        .prob-fill {
            height: 100%;
            border-radius: 5px;
            transition: width 0.3s ease;
        }
        .prob-pct {
            width: 42px;
            text-align: right;
            font-family: 'JetBrains Mono', monospace;
            color: #4dc9ff;
            font-size: 0.75rem;
        }

        /* ── Feature table ── */
        .feat-table-row {
            display: flex;
            justify-content: space-between;
            padding: 0.35rem 0.5rem;
            border-bottom: 1px solid rgba(255,255,255,0.04);
            font-size: 0.8rem;
        }
        .feat-table-row:hover { background: rgba(0,200,255,0.05); }
        </style>
        """,
        unsafe_allow_html=True,
    )
