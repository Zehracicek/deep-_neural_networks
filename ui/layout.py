"""SOC paneli düzen bileşenleri (KPI, durum çubuğu, örnek kartları)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from preprocess_nsl_kdd import CLASS_NAMES

# Sınıf renkleri ve ikonlar
SINIF_STIL: dict[str, dict[str, str]] = {
    "Normal": {"renk": "#00e87a", "ikon": "✅", "bg": "rgba(0,232,122,0.12)"},
    "DoS": {"renk": "#ff6644", "ikon": "🚨", "bg": "rgba(255,102,68,0.12)"},
    "Probe": {"renk": "#ffb020", "ikon": "🔍", "bg": "rgba(255,176,32,0.12)"},
    "R2L": {"renk": "#ff4466", "ikon": "⚠️", "bg": "rgba(255,68,102,0.12)"},
    "U2R": {"renk": "#ff2244", "ikon": "🔥", "bg": "rgba(255,34,68,0.12)"},
}

SINIF_BAR_RENK: dict[str, str] = {ad: st["renk"] for ad, st in SINIF_STIL.items()}


def sinif_rozet_html(sinif_adi: str) -> str:
    stil = SINIF_STIL.get(sinif_adi, {"renk": "#4dc9ff", "ikon": "•", "bg": "rgba(77,201,255,0.1)"})
    return (
        f'<span class="class-badge" style="color:{stil["renk"]};background:{stil["bg"]};'
        f'border:1px solid {stil["renk"]}55;">{stil["ikon"]} {sinif_adi}</span>'
    )


def durum_cubugu_goster(*, model_hazir: bool = True, tahmin_sayisi: int = 0) -> None:
    """Üst SOC durum şeridi."""
    simdi = datetime.now().strftime("%H:%M:%S")
    durum = "ÇEVRİMİÇİ" if model_hazir else "YÜKLENİYOR"
    renk = "#00ff88" if model_hazir else "#ffb020"
    st.markdown(
        f"""
        <div class="soc-status-strip">
            <div class="soc-status-left">
                <span class="soc-pulse" style="background:{renk};"></span>
                <span style="color:{renk};font-weight:700;font-family:monospace;">{durum}</span>
                <span class="soc-status-sep">|</span>
                <span class="soc-status-meta">Hibrit LSTM+DNN · 5 sınıf</span>
            </div>
            <div class="soc-status-right">
                <span class="soc-status-meta">🕐 {simdi}</span>
                <span class="soc-status-sep">|</span>
                <span class="soc-status-meta">📡 {tahmin_sayisi} tarama</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_seridi_goster(
    egitim_sayisi: int,
    test_sayisi: int,
    ozellik_sayisi: int,
    pencere: int,
    saldiri_orani: float,
) -> None:
    """Ana KPI kartları."""
    k1, k2, k3, k4, k5 = st.columns(5)
    kartlar = [
        (k1, f"{egitim_sayisi:,}", "Eğitim (SMOTE)"),
        (k2, f"{test_sayisi:,}", "Test örnekleri"),
        (k3, str(ozellik_sayisi), "Özellik boyutu"),
        (k4, str(pencere), "LSTM penceresi"),
        (k5, f"%{saldiri_orani:.1f}", "Test saldırı oranı"),
    ]
    for col, deger, etiket in kartlar:
        with col:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-value">{deger}</div>'
                f'<div class="kpi-label">{etiket}</div></div>',
                unsafe_allow_html=True,
            )


def ornek_ozet_karti(
    *,
    ornek_no: int,
    gercek_sinif: str,
    protokol: str | None,
    servis: str | None,
    bayrak: str | None,
    aktif_ozellik: int,
    toplam_ozellik: int,
) -> None:
    """Seçili akış özeti."""
    proto = protokol or "—"
    srv = servis or "—"
    flg = bayrak or "—"
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-card-title">📋 Seçili ağ akışı</div>
            <div style="display:flex;flex-wrap:wrap;gap:0.5rem;align-items:center;margin:0.5rem 0;">
                {sinif_rozet_html(gercek_sinif)}
                <span class="meta-chip"># {ornek_no}</span>
                <span class="meta-chip">🌐 {proto}</span>
                <span class="meta-chip">📡 {srv}</span>
                <span class="meta-chip">🚩 {flg}</span>
            </div>
            <div class="meta-line">
                Aktif sinyal: <b>{aktif_ozellik}</b> / {toplam_ozellik} özellik
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kenar_cubugu_dagilim_grafigi(dagilim: dict[str, int]) -> None:
    """Sidebar için mini sınıf dağılımı."""
    if not dagilim:
        return
    df = pd.DataFrame({"Sınıf": list(dagilim.keys()), "Adet": list(dagilim.values())})
    st.bar_chart(df.set_index("Sınıf"), color="#4dc9ff", height=140)


def kenar_cubugu_goster(veri: dict, *, model_hazir: bool = True) -> None:
    """Geliştirilmiş kenar çubuğu."""
    sinif_adlari = veri.get("sinif_adlari", CLASS_NAMES)
    y_test = veri["y_test"]
    ozellik_sayisi = veri["X_test"].shape[1]
    pencere = veri["pencere"]

    st.markdown(
        '<div class="sidebar-header"><h2>🛡️ DNN-IDS</h2>'
        '<p class="sidebar-sub">Security Operations Center</p></div>',
        unsafe_allow_html=True,
    )

    if model_hazir:
        st.markdown('<div class="sidebar-online">● Model aktif</div>', unsafe_allow_html=True)
    else:
        st.warning("Model yükleniyor…")

    st.markdown("### 📊 Özet")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Eğitim", f"{len(veri['y_egitim']):,}")
    with c2:
        st.metric("Test", f"{len(y_test):,}")
    st.metric("Özellik", ozellik_sayisi)
    st.metric("Pencere", pencere)

    st.divider()
    st.markdown("### 🎯 Test dağılımı")
    from preprocess_nsl_kdd import class_distribution

    dag = class_distribution(y_test, sinif_adlari)
    kenar_cubugu_dagilim_grafigi(dag)
    for ad in sinif_adlari:
        st.caption(f"{SINIF_STIL.get(ad, {}).get('ikon', '•')} {ad}: **{dag[ad]:,}**")

    st.divider()
    st.markdown("### 🧠 Mimari")
    st.markdown(
        """
        <div class="info-box">
        <b>DNN</b> statik vektör<br>
        <b>LSTM</b> paket dizisi<br>
        <b>SMOTE</b> + ağırlık<br>
        <b>5 sınıf</b> softmax
        </div>
        """,
        unsafe_allow_html=True,
    )

    if veri.get("smote_once") and veri.get("smote_sonra"):
        with st.expander("📈 SMOTE etkisi", expanded=False):
            for ad in veri["smote_once"]:
                once = veri["smote_once"][ad]
                sonra = veri["smote_sonra"].get(ad, 0)
                st.caption(f"**{ad}:** {once:,} → {sonra:,}")
