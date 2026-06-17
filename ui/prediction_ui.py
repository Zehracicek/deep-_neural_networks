"""Çok sınıflı tahmin sonuçları — gelişmiş SOC kartları."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from preprocess_nsl_kdd import CLASS_NAMES
from ui.charts import _soc_layout
from ui.layout import SINIF_BAR_RENK, SINIF_STIL, sinif_rozet_html

SINIF_RENKLERI = {
    "Normal": ("#00e87a", "pred-card-normal", "✅ NORMAL TRAFİK"),
    "DoS": ("#ff6644", "pred-card-attack", "🚨 DoS SALDIRISI"),
    "Probe": ("#ffb020", "pred-card-attack", "🔍 PROBE (KEŞİF)"),
    "R2L": ("#ff4466", "pred-card-attack", "🚨 R2L SALDIRISI"),
    "U2R": ("#ff2244", "pred-card-attack", "🚨 U2R SALDIRISI"),
}

def ust_ozellikler_placeholder(
    statik_vektor: np.ndarray,
    ozellik_adlari: np.ndarray,
    *,
    ust_k: int = 10,
) -> list[tuple[str, float]]:
    v = np.asarray(statik_vektor).ravel()
    adlar = list(ozellik_adlari)
    skorlar = np.abs(v)
    sirali = np.argsort(skorlar)[::-1][:ust_k]
    return [(adlar[i], float(v[i])) for i in sirali]


def _olasilik_pasta(olasiliklar: dict[str, float], tahmin: str) -> go.Figure:
    etiketler = list(olasiliklar.keys())
    degerler = [olasiliklar[k] for k in etiketler]
    renkler = [SINIF_BAR_RENK.get(k, "#4dc9ff") for k in etiketler]
    fig = go.Figure(
        go.Pie(
            labels=etiketler,
            values=degerler,
            hole=0.45,
            marker=dict(colors=renkler),
            textinfo="percent+label",
            textfont_size=11,
            pull=[0.05 if k == tahmin else 0 for k in etiketler],
        )
    )
    fig.update_layout(
        title="Sınıf olasılıkları",
        showlegend=False,
        margin=dict(l=10, r=10, t=36, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9eb8d8"),
        height=280,
    )
    return fig


def _olasilik_barlari_html(olasiliklar: dict[str, float], tahmin: str) -> str:
    satirlar = []
    for ad, p in sorted(olasiliklar.items(), key=lambda x: -x[1]):
        renk = SINIF_BAR_RENK.get(ad, "#4dc9ff")
        kalin = "font-weight:800;" if ad == tahmin else ""
        satirlar.append(
            f'<div class="prob-row">'
            f'<span class="prob-label" style="{kalin}">{ad}</span>'
            f'<div class="prob-track"><div class="prob-fill" style="width:{p*100:.1f}%;background:{renk};"></div></div>'
            f'<span class="prob-pct">{p:.1%}</span></div>'
        )
    return "".join(satirlar)


def tahmin_yap(
    model: Any,
    statik_girdi: np.ndarray,
    dizi_girdi: np.ndarray,
    *,
    sinif_adlari: tuple[str, ...] = CLASS_NAMES,
) -> dict[str, Any] | None:
    try:
        proba = model.predict([statik_girdi, dizi_girdi], verbose=0)[0]
        sinif_id = int(np.argmax(proba))
        sinif_adi = sinif_adlari[sinif_id]
        guven = float(proba[sinif_id]) * 100.0
        saldiri_olasiligi = float(1.0 - proba[0])
        tehdit_etiketi, tehdit_renk = tehdit_seviyesi_hesapla(saldiri_olasiligi)
        return {
            "sinif": sinif_id,
            "sinif_adi": sinif_adi,
            "olasiliklar": {sinif_adlari[i]: float(proba[i]) for i in range(len(sinif_adlari))},
            "guven": guven,
            "saldiri_olasiligi": saldiri_olasiligi,
            "tehdit_etiketi": tehdit_etiketi,
            "tehdit_renk": tehdit_renk,
        }
    except Exception as hata:
        st.error(f"Tahmin hatası: {hata}")
        return None


def tehdit_seviyesi_hesapla(saldiri_olasiligi: float) -> tuple[str, str]:
    if saldiri_olasiligi < 0.35:
        return "DÜŞÜK", "#00ff88"
    if saldiri_olasiligi < 0.65:
        return "ORTA", "#ffb020"
    return "YÜKSEK", "#ff3344"


def tahmin_gecmisine_ekle(oturum: Any, sonuc: dict[str, Any]) -> None:
    if "tahmin_gecmisi" not in oturum:
        oturum.tahmin_gecmisi = []
    oturum.tahmin_gecmisi.append(
        {
            "saldiri_olasiligi": sonuc["saldiri_olasiligi"],
            "etiket": sonuc["sinif_adi"],
            "guven": sonuc["guven"],
        }
    )
    oturum.tahmin_gecmisi = oturum.tahmin_gecmisi[-30:]


def tahmin_sonucu_goster(
    sonuc: dict[str, Any],
    *,
    ust_ozellikler: list[tuple[str, float]] | None = None,
) -> None:
    sinif_adi = sonuc["sinif_adi"]
    _, kart, baslik = SINIF_RENKLERI.get(sinif_adi, ("#4dc9ff", "pred-card-normal", sinif_adi))

    sol, sag = st.columns([1.1, 1])
    with sol:
        st.markdown(
            f"""
            <div class="{kart}">
                <h2>{baslik}</h2>
                <div class="pred-meta" style="margin-top:0.5rem;">
                    {sinif_rozet_html(sinif_adi)}
                </div>
                <span class="threat-pill" style="background:{sonuc['tehdit_renk']}22;
                    color:{sonuc['tehdit_renk']};border:1px solid {sonuc['tehdit_renk']}66;
                    margin-top:0.65rem;">
                    TEHDİT: {sonuc['tehdit_etiketi']}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Güven", f"{sonuc['guven']:.1f}%")
        with m2:
            st.metric("P(saldırı)", f"{sonuc['saldiri_olasiligi']:.1%}")
        with m3:
            st.metric("Sınıf ID", sonuc["sinif"])
        st.progress(
            min(max(sonuc["saldiri_olasiligi"], 0.0), 1.0),
            text=f"Tehdit skoru: {sonuc['saldiri_olasiligi']:.3f}",
        )

    with sag:
        st.plotly_chart(
            _olasilik_pasta(sonuc["olasiliklar"], sinif_adi),
            use_container_width=True,
        )

    st.markdown('<div class="panel-title">SINIF OLASILIKLARI</div>', unsafe_allow_html=True)
    st.markdown(_olasilik_barlari_html(sonuc["olasiliklar"], sinif_adi), unsafe_allow_html=True)

    if ust_ozellikler:
        st.markdown('<div class="panel-title">ÖNEMLİ ÖZELLİKLER (geçici skor)</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1.2, 1])
        with c1:
            for ad, deger in ust_ozellikler[:6]:
                st.markdown(
                    f'<div class="feat-table-row"><span style="color:#9eb8d8">{ad[:32]}</span>'
                    f'<span style="color:#4dc9ff;font-family:monospace">{deger:.4f}</span></div>',
                    unsafe_allow_html=True,
                )
        with c2:
            adlar = [o[0][:20] for o in ust_ozellikler]
            degerler = [abs(o[1]) for o in ust_ozellikler]
            fig = go.Figure(go.Bar(x=degerler, y=adlar, orientation="h", marker_color="#4dc9ff"))
            fig.update_layout(
                margin=dict(l=4, r=4, t=8, b=4),
                height=220,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,16,28,0.5)",
                font=dict(color="#9eb8d8", size=10),
            )
            st.plotly_chart(fig, use_container_width=True)


run_prediction = tahmin_yap
render_prediction_result = tahmin_sonucu_goster
append_prediction_history = tahmin_gecmisine_ekle
