"""Model açıklanabilirliği — SHAP (isteğe bağlı) veya geçici özellik skoru."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.charts import _soc_layout
from ui.prediction_ui import ust_ozellikler_placeholder

# SHAP opsiyonel (ağır bağımlılık); yoksa placeholder kullanılır
try:
    import shap

    _SHAP_MEVCUT = True
except ImportError:
    _SHAP_MEVCUT = False


def _placeholder_bar(ozellikler: list[tuple[str, float]]) -> go.Figure:
    adlar = [o[0][:28] for o in ozellikler]
    degerler = [o[1] for o in ozellikler]
    fig = go.Figure(go.Bar(x=degerler, y=adlar, orientation="h", marker_color="#4dc9ff"))
    fig.update_layout(
        title="Önemli özellikler (geçici skor — SHAP yakında)",
        yaxis=dict(autorange="reversed"),
    )
    return _soc_layout(fig, height=260)


def shap_aciklamasi_goster(
    model: Any,
    X_train: np.ndarray,
    ozellik_vektoru: np.ndarray,
    ozellik_adlari: np.ndarray,
    *,
    dizi_girdi: np.ndarray | None = None,
) -> None:
    st.markdown('<div class="panel-title">MODEL AÇIKLANABİLİRLİĞİ</div>', unsafe_allow_html=True)

    placeholder = ust_ozellikler_placeholder(ozellik_vektoru, ozellik_adlari, ust_k=10)
    st.plotly_chart(_placeholder_bar(placeholder), use_container_width=True)
    st.caption(
        "Geçici skor: ön işlenmiş özelliklerin mutlak büyüklüğü. "
        "Tam SHAP hibrit modele eklenecek."
    )

    if not _SHAP_MEVCUT or dizi_girdi is None:
        return

    try:
        bg_idx = np.random.default_rng(42).choice(
            len(X_train), size=min(50, len(X_train)), replace=False
        )
        arka_plan = X_train[bg_idx]
        explainer = shap.DeepExplainer(model, [arka_plan, arka_plan])  # may fail on multi-input
        st.info("Çok girdili hibrit model için SHAP deneysel aşamada; placeholder kullanılıyor.")
    except Exception:
        pass
