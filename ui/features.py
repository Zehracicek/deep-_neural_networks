"""Gruplanmış, yalnızca aktif (sıfır olmayan) özellik gösterimi."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import streamlit as st

# Özellik bölümleri (Türkçe başlıklar)
OZELLIK_BOLUMLERI: tuple[tuple[str, str, str], ...] = (
    ("ag", "🌐 Ağ Özellikleri", "ag"),
    ("protokol", "🔐 Protokol Özellikleri", "protokol"),
    ("servis", "📡 Servis Özellikleri", "servis"),
    ("bayrak", "🚩 Bayrak Özellikleri", "bayrak"),
)

SAYISAL_ESIK = 0.01
ONEHOT_ESIK = 0.5


@dataclass(frozen=True)
class AktifOzellik:
    """Gösterilecek tek bir aktif özellik."""

    ad: str
    gorunen_ad: str
    deger: float
    bolum: str


def _gorunen_ad(ham_ad: str) -> str:
    for on_ek in ("protocol_type_", "service_", "flag_"):
        if ham_ad.startswith(on_ek):
            return ham_ad[len(on_ek) :].replace("_", " ").upper()
    return ham_ad.replace("_", " ").title()


def _bolum_belirle(ad: str, sayisal_sutunlar: set[str]) -> str:
    if ad in sayisal_sutunlar:
        return "ag"
    if ad.startswith("protocol_type_"):
        return "protokol"
    if ad.startswith("service_"):
        return "servis"
    if ad.startswith("flag_"):
        return "bayrak"
    return "ag"


def aktif_ozellik_mi(ad: str, deger: float, sayisal_sutunlar: set[str]) -> bool:
    """One-hot: >0.5; ölçeklenmiş sayısal: |değer| > eşik."""
    if ad in sayisal_sutunlar:
        return abs(float(deger)) > SAYISAL_ESIK
    return float(deger) > ONEHOT_ESIK


def aktif_ozellikleri_cikar(
    ozellik_vektoru: np.ndarray,
    ozellik_adlari: np.ndarray,
    sayisal_sutunlar: list[str],
) -> list[AktifOzellik]:
    sayisal = set(sayisal_sutunlar)
    adlar = list(ozellik_adlari)
    vektor = np.asarray(ozellik_vektoru).ravel()
    sonuc: list[AktifOzellik] = []

    for i, ad in enumerate(adlar):
        deger = float(vektor[i])
        if not aktif_ozellik_mi(ad, deger, sayisal):
            continue
        sonuc.append(
            AktifOzellik(
                ad=ad,
                gorunen_ad=_gorunen_ad(ad),
                deger=deger,
                bolum=_bolum_belirle(ad, sayisal),
            )
        )
    return sonuc


def bolume_gore_grupla(aktifler: list[AktifOzellik]) -> dict[str, list[AktifOzellik]]:
    gruplu: dict[str, list[AktifOzellik]] = {k: [] for k, _, _ in OZELLIK_BOLUMLERI}
    for oz in aktifler:
        gruplu.setdefault(oz.bolum, []).append(oz)
    return gruplu


def _ozellik_cipleri_goster(ozellikler: list[AktifOzellik]) -> None:
    if not ozellikler:
        st.caption("_Bu bölümde aktif özellik yok._")
        return
    cipler = []
    for o in sorted(ozellikler, key=lambda x: abs(x.deger), reverse=True):
        cipler.append(
            f'<span class="feat-chip">'
            f'<span class="fname">{o.gorunen_ad}</span>'
            f'<span class="fval">{o.deger:.4f}</span>'
            f"</span>"
        )
    st.markdown("".join(cipler), unsafe_allow_html=True)


def ozellik_bolumlerini_goster(
    ozellik_vektoru: np.ndarray,
    ozellik_adlari: np.ndarray,
    sayisal_sutunlar: list[str],
    *,
    toplam_boyut: int,
) -> None:
    """Aktif özellikleri gruplu genişletilebilir panellerde göster."""
    aktifler = aktif_ozellikleri_cikar(ozellik_vektoru, ozellik_adlari, sayisal_sutunlar)
    gruplu = bolume_gore_grupla(aktifler)

    st.markdown(
        f'<div class="panel-title">AKTİF SİNYALLER — {len(aktifler)} / {toplam_boyut} özellik</div>',
        unsafe_allow_html=True,
    )

    for anahtar, baslik, _ in OZELLIK_BOLUMLERI:
        bolum_oz = gruplu.get(anahtar, [])
        acik = anahtar == "ag" and len(bolum_oz) > 0
        with st.expander(f"{baslik} ({len(bolum_oz)})", expanded=acik):
            _ozellik_cipleri_goster(bolum_oz)


# Geriye uyumluluk (eski İngilizce import adları)
render_feature_sections = ozellik_bolumlerini_goster

__all__ = ["ozellik_bolumlerini_goster", "render_feature_sections"]
