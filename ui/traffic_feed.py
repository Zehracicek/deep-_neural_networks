"""Canlı trafik izleme — renkli durum satırları."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from preprocess_nsl_kdd import CLASS_NAMES
from ui.layout import SINIF_STIL

PROTOKOLLER = ("TCP", "UDP", "ICMP")


def _protokol_satirdan(satir: pd.Series | None, rng: np.random.Generator) -> str:
    if satir is not None and "protocol_type" in satir.index:
        return str(satir["protocol_type"]).upper()
    return str(rng.choice(PROTOKOLLER))


def trafik_olayi_ekle(
    oturum: Any,
    *,
    protokol: str,
    sinif_adi: str = "Normal",
    max_satir: int = 15,
) -> None:
    if "trafik_gunlugu" not in oturum:
        oturum.trafik_gunlugu = []
    ikon = SINIF_STIL.get(sinif_adi, {}).get("ikon", "•")
    oturum.trafik_gunlugu.insert(
        0,
        {
            "Zaman": datetime.now().strftime("%H:%M:%S"),
            "Protokol": protokol,
            "Sınıf": f"{ikon} {sinif_adi}",
            "Tehdit": "Hayır" if sinif_adi == "Normal" else "Evet",
        },
    )
    oturum.trafik_gunlugu = oturum.trafik_gunlugu[:max_satir]


def trafik_patlamasi_simule_et(
    oturum: Any,
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    sinif_adlari: tuple[str, ...] = CLASS_NAMES,
    *,
    olay_sayisi: int = 6,
) -> None:
    rng = np.random.default_rng()
    indeksler = rng.integers(0, len(y_test), size=olay_sayisi)
    for idx in indeksler:
        satir = df_test.iloc[int(idx)] if len(df_test) else None
        sinif = sinif_adlari[int(y_test[int(idx)])]
        trafik_olayi_ekle(
            oturum,
            protokol=_protokol_satirdan(satir, rng),
            sinif_adi=sinif,
        )


@st.fragment(run_every=5)
def canli_trafik_otomatik(
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    sinif_adlari: tuple[str, ...] = CLASS_NAMES,
) -> None:
    if not st.session_state.get("otomatik_trafik", False):
        return
    rng = np.random.default_rng()
    idx = int(rng.integers(0, len(y_test)))
    satir = df_test.iloc[idx]
    trafik_olayi_ekle(
        st.session_state,
        protokol=_protokol_satirdan(satir, rng),
        sinif_adi=sinif_adlari[int(y_test[idx])],
    )


def trafik_izleyici_goster(
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    sinif_adlari: tuple[str, ...] = CLASS_NAMES,
) -> None:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-card-title">📡 Canlı trafik izleyici</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("📡 Simüle et", use_container_width=True, key="trafik_patlamasi"):
            trafik_patlamasi_simule_et(
                st.session_state, df_test, y_test, sinif_adlari, olay_sayisi=6
            )
    with c2:
        st.session_state.otomatik_trafik = st.toggle(
            "Otomatik (5 sn)",
            value=st.session_state.get("otomatik_trafik", False),
            key="otomatik_trafik_anahtar",
        )
    with c3:
        if st.button("🗑️ Temizle", use_container_width=True, key="trafik_temizle"):
            st.session_state.trafik_gunlugu = []

    if st.session_state.get("otomatik_trafik", False):
        canli_trafik_otomatik(df_test, y_test, sinif_adlari)

    gunluk = st.session_state.get("trafik_gunlugu", [])
    if not gunluk:
        st.caption("_Tahmin yapın veya simülasyon başlatın — kayıtlar burada görünür._")
        return

    df = pd.DataFrame(gunluk)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=min(35 * len(df) + 38, 280),
        column_config={
            "Zaman": st.column_config.TextColumn("Zaman", width="small"),
            "Protokol": st.column_config.TextColumn("Protokol", width="small"),
            "Sınıf": st.column_config.TextColumn("Sınıf", width="medium"),
            "Tehdit": st.column_config.TextColumn("Tehdit", width="small"),
        },
    )


render_traffic_monitor = trafik_izleyici_goster
append_traffic_event = trafik_olayi_ekle
