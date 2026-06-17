"""Plotly grafikleri — gelişmiş SOC analitik."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from preprocess_nsl_kdd import CLASS_NAMES
RENKLER = ["#00c878", "#ff4466", "#4dc9ff", "#ffb020", "#cc44ff"]


def _soc_layout(fig: go.Figure, height: int = 220) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,16,28,0.6)",
        font=dict(color="#9eb8d8", size=11),
        title_font=dict(color="#c8e0ff", size=13),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9eb8d8")),
    )
    fig.update_xaxes(gridcolor="rgba(80,120,180,0.12)", zerolinecolor="rgba(80,120,180,0.12)")
    fig.update_yaxes(gridcolor="rgba(80,120,180,0.12)", zerolinecolor="rgba(80,120,180,0.12)")
    return fig


def _sinif_sayilari(y: np.ndarray, sinif_adlari: tuple[str, ...]) -> list[int]:
    y = np.asarray(y).astype(int).ravel()
    return [int((y == i).sum()) for i in range(len(sinif_adlari))]


def plot_class_distribution(
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    sinif_adlari: tuple[str, ...] = CLASS_NAMES,
) -> go.Figure:
    etiketler = list(sinif_adlari)
    egitim = _sinif_sayilari(y_train, sinif_adlari)
    test = _sinif_sayilari(y_test, sinif_adlari)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Eğitim", x=etiketler, y=egitim, marker=dict(color=RENKLER[: len(etiketler)]))
    )
    fig.add_trace(
        go.Bar(
            name="Test",
            x=etiketler,
            y=test,
            marker=dict(color=RENKLER[: len(etiketler)], opacity=0.55),
        )
    )
    fig.update_layout(barmode="group", title="Sınıf dağılımı")
    return _soc_layout(fig, height=250)


def plot_smote_comparison(
    smote_before: dict[str, int],
    smote_after: dict[str, int],
) -> go.Figure:
    siniflar = list(smote_before.keys())
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="SMOTE öncesi",
            x=siniflar,
            y=[smote_before[s] for s in siniflar],
            marker_color="#6b8cae",
        )
    )
    fig.add_trace(
        go.Bar(
            name="SMOTE sonrası",
            x=siniflar,
            y=[smote_after.get(s, 0) for s in siniflar],
            marker_color="#4dc9ff",
        )
    )
    fig.update_layout(barmode="group", title="SMOTE dengeleme etkisi")
    return _soc_layout(fig, height=240)


def plot_confidence_history(gecmis: list[dict[str, Any]]) -> go.Figure:
    if not gecmis:
        fig = go.Figure()
        fig.add_annotation(
            text="Tahmin yaptıkça dolacak",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#6b8cae", size=12),
        )
        fig.update_layout(title="Tarama geçmişi")
        return _soc_layout(fig, height=200)

    df = pd.DataFrame(gecmis)
    df["tarama"] = range(1, len(df) + 1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["tarama"],
            y=df["saldiri_olasiligi"],
            mode="lines+markers",
            name="Tehdit skoru",
            line=dict(color="#ff4466", width=2),
            marker=dict(size=7),
        )
    )
    if "guven" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["tarama"],
                y=df["guven"] / 100.0,
                mode="lines",
                name="Güven",
                line=dict(color="#4dc9ff", dash="dot"),
            )
        )
    fig.add_hline(y=0.5, line_dash="dot", line_color="#ffb020")
    fig.update_layout(
        title="Tahmin geçmişi",
        xaxis_title="Tarama",
        yaxis_title="Skor",
    )
    return _soc_layout(fig, height=220)


def plot_traffic_simulation(tohum: int | None = None) -> go.Figure:
    rng = np.random.default_rng(tohum)
    simdi = datetime.now()
    zamanlar = [simdi - timedelta(seconds=30 - i * 3) for i in range(11)]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=zamanlar,
            y=rng.integers(40, 120, 11),
            name="Normal",
            mode="lines+markers",
            fill="tozeroy",
            line=dict(color="#00c878"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=zamanlar,
            y=rng.integers(0, 35, 11),
            name="Şüpheli",
            mode="lines+markers",
            fill="tozeroy",
            line=dict(color="#ff4466"),
        )
    )
    fig.update_layout(title="Canlı trafik simülasyonu", xaxis_title="Zaman")
    return _soc_layout(fig, height=220)


def render_analytics_panel(
    y_train: np.ndarray,
    y_test: np.ndarray,
    tahmin_gecmisi: list[dict[str, Any]],
    sim_tohum: int,
    *,
    sinif_adlari: tuple[str, ...] = CLASS_NAMES,
    smote_before: dict[str, int] | None = None,
    smote_after: dict[str, int] | None = None,
) -> None:
    st.markdown("#### 📊 SOC analitik paneli")

    if smote_before and smote_after:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_smote_comparison(smote_before, smote_after), use_container_width=True)
        with c2:
            df = pd.DataFrame(
                {
                    "Sınıf": list(smote_before.keys()),
                    "Önce": list(smote_before.values()),
                    "Sonra": [smote_after.get(k, 0) for k in smote_before],
                }
            )
            df["Artış"] = df["Sonra"] - df["Önce"]
            st.dataframe(df, use_container_width=True, hide_index=True)

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(
            plot_class_distribution(y_train, y_test, sinif_adlari=sinif_adlari),
            use_container_width=True,
        )
    with g2:
        st.plotly_chart(plot_confidence_history(tahmin_gecmisi), use_container_width=True)

    st.plotly_chart(plot_traffic_simulation(sim_tohum), use_container_width=True)
