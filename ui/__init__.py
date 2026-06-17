"""DNN-IDS Streamlit arayüz modülleri (çok sınıflı hibrit model)."""

from ui.features import aktif_ozellikleri_cikar, ozellik_bolumlerini_goster, render_feature_sections
from ui.layout import (
    durum_cubugu_goster,
    kenar_cubugu_goster,
    kpi_seridi_goster,
    ornek_ozet_karti,
)
from ui.prediction_ui import (
    tahmin_gecmisine_ekle,
    tahmin_sonucu_goster,
    tahmin_yap,
    ust_ozellikler_placeholder,
)
from ui.shap_explain import shap_aciklamasi_goster
from ui.traffic_feed import trafik_izleyici_goster, trafik_olayi_ekle

__all__ = [
    "ozellik_bolumlerini_goster",
    "render_feature_sections",
    "tahmin_yap",
    "tahmin_sonucu_goster",
    "tahmin_gecmisine_ekle",
    "ust_ozellikler_placeholder",
    "shap_aciklamasi_goster",
    "trafik_izleyici_goster",
    "trafik_olayi_ekle",
]
