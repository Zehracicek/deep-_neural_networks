"""
NSL-KDD Hibrit LSTM+DNN IDS — Streamlit SOC paneli (5 sınıf).

Çalıştırma: streamlit run app.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import streamlit as st

from hybrid_model import train_hybrid_model
from preprocess_nsl_kdd import (
    class_distribution,
    load_raw_frames,
    multiclass_labels,
    prepare_hybrid_dataset,
)
from ui.charts import render_analytics_panel
from ui.features import aktif_ozellikleri_cikar, ozellik_bolumlerini_goster
from ui.layout import (
    durum_cubugu_goster,
    kenar_cubugu_goster,
    kpi_seridi_goster,
    ornek_ozet_karti,
    sinif_rozet_html,
)
from ui.theme import inject_soc_theme

try:
    from ui import (
        shap_aciklamasi_goster,
        tahmin_gecmisine_ekle,
        tahmin_sonucu_goster,
        tahmin_yap,
        trafik_izleyici_goster,
        trafik_olayi_ekle,
        ust_ozellikler_placeholder,
    )
except ImportError:
    from ui.prediction_ui import (
        append_prediction_history as tahmin_gecmisine_ekle,
        render_prediction_result as tahmin_sonucu_goster,
        run_prediction as tahmin_yap,
        ust_ozellikler_placeholder,
    )
    from ui.shap_explain import render_shap_explanation as shap_aciklamasi_goster
    from ui.traffic_feed import (
        append_traffic_event as trafik_olayi_ekle,
        render_traffic_monitor as trafik_izleyici_goster,
    )

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="DNN-IDS | SOC Paneli",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Veri yükleniyor (SMOTE + LSTM dizileri)…")
def veri_yukle():
    try:
        hazir = prepare_hybrid_dataset(apply_smote=True)
        _, df_test = load_raw_frames()
        return {
            "hazir": hazir,
            "df_test": df_test,
            "y_test_ham": multiclass_labels(df_test),
            "X_egitim": hazir.X_train,
            "X_seq_egitim": hazir.X_seq_train,
            "y_egitim": hazir.y_train,
            "X_test": hazir.X_test,
            "X_seq_test": hazir.X_seq_test,
            "y_test": hazir.y_test,
            "ozellik_adlari": hazir.feature_names,
            "sayisal_sutunlar": hazir.numerical_columns,
            "sinif_adlari": hazir.class_names,
            "pencere": hazir.window_size,
            "gecerli_test_indeksleri": hazir.valid_test_indices,
            "smote_once": hazir.smote_before,
            "smote_sonra": hazir.smote_after,
        }
    except Exception as hata:
        st.error(f"Veri yükleme hatası: {hata}")
        st.stop()


@st.cache_resource(show_spinner="Hibrit model eğitiliyor (ilk açılış uzun sürebilir)…")
def model_egit(X_egitim, X_seq_egitim, y_egitim, pencere: int):
    try:
        model, _ = train_hybrid_model(
            X_egitim,
            X_seq_egitim,
            y_egitim,
            epochs=10,
            validation_split=0.2,
            batch_size=512,
            early_stopping_patience=3,
            verbose=0,
            window_size=pencere,
        )
        return model
    except Exception as hata:
        st.error(f"Model eğitim hatası: {hata}")
        st.stop()


def _gercek_etiket_karsilastir(gercek_sinif: int, sonuc: dict, sinif_adlari: tuple) -> None:
    gercek_ad = sinif_adlari[gercek_sinif]
    tahmin_ad = sonuc["sinif_adi"]
    dogru = gercek_sinif == sonuc["sinif"]
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-card-title">✓ Doğruluk kontrolü</div>
            <div style="display:flex;gap:1rem;align-items:center;flex-wrap:wrap;">
                <span>Gerçek: {sinif_rozet_html(gercek_ad)}</span>
                <span>Tahmin: {sinif_rozet_html(tahmin_ad)}</span>
                <span class="{'match-ok' if dogru else 'match-bad'}" style="padding:0.35rem 0.75rem;border-radius:6px;">
                    {'✅ DOĞRU' if dogru else '❌ YANLIŞ'}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _tahmin_akisi(model, veri, statik, dizi, *, gercek_sinif, ham_satir):
    sonuc = tahmin_yap(model, statik, dizi, sinif_adlari=veri["sinif_adlari"])
    if not sonuc:
        return

    ust = ust_ozellikler_placeholder(statik.ravel(), veri["ozellik_adlari"])
    tahmin_gecmisine_ekle(st.session_state, sonuc)
    tahmin_sonucu_goster(sonuc, ust_ozellikler=ust)

    if gercek_sinif is not None:
        _gercek_etiket_karsilastir(gercek_sinif, sonuc, veri["sinif_adlari"])

    with st.expander("🔬 Model açıklanabilirliği", expanded=False):
        shap_aciklamasi_goster(
            model, veri["X_egitim"], statik.ravel(), veri["ozellik_adlari"], dizi_girdi=dizi
        )

    protokol = "TCP"
    if ham_satir is not None and "protocol_type" in ham_satir.index:
        protokol = str(ham_satir["protocol_type"]).upper()
    trafik_olayi_ekle(st.session_state, protokol=protokol, sinif_adi=sonuc["sinif_adi"])


def _ham_satir_alanlari(satir: pd.Series) -> tuple[str | None, str | None, str | None]:
    proto = str(satir["protocol_type"]).upper() if "protocol_type" in satir.index else None
    servis = str(satir["service"]) if "service" in satir.index else None
    bayrak = str(satir["flag"]) if "flag" in satir.index else None
    return proto, servis, bayrak


def main() -> None:
    inject_soc_theme()

    st.markdown(
        """
        <div class="hero-header">
            <h1>🛡️ SOC — Ağ Saldırı Tespit Konsolu</h1>
            <p>Hibrit LSTM + DNN · NSL-KDD · Normal · DoS · Probe · R2L · U2R</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    veri = veri_yukle()
    sinif_adlari = veri["sinif_adlari"]
    X_test, X_seq_test, y_test = veri["X_test"], veri["X_seq_test"], veri["y_test"]
    df_test, gecerli_idx = veri["df_test"], veri["gecerli_test_indeksleri"]
    ozellik_sayisi = X_test.shape[1]
    pencere = veri["pencere"]

    model = model_egit(veri["X_egitim"], veri["X_seq_egitim"], veri["y_egitim"], pencere)

    if "tahmin_gecmisi" not in st.session_state:
        st.session_state.tahmin_gecmisi = []
    if "trafik_gunlugu" not in st.session_state:
        st.session_state.trafik_gunlugu = []
    if "sim_tohum" not in st.session_state:
        st.session_state.sim_tohum = 42

    with st.sidebar:
        kenar_cubugu_goster(veri, model_hazir=True)

    tahmin_sayisi = len(st.session_state.tahmin_gecmisi)
    durum_cubugu_goster(model_hazir=True, tahmin_sayisi=tahmin_sayisi)

    dag_test = class_distribution(y_test, sinif_adlari)
    saldiri_oran = (1 - dag_test.get("Normal", 0) / max(len(y_test), 1)) * 100
    kpi_seridi_goster(len(veri["y_egitim"]), len(y_test), ozellik_sayisi, pencere, saldiri_oran)

    st.markdown(
        '<div class="glass-card">✅ <b>Sistem operasyonel</b> — SMOTE + hibrit model hazır.</div>',
        unsafe_allow_html=True,
    )

    sekme_rastgele, sekme_manuel, sekme_model = st.tabs(
        ["🎲 Rastgele Örnek Testi", "🖊️ Manuel Veri Girişi", "📈 Model Detayları"]
    )

    # ── Sekme 1: Rastgele örnek + trafik (iki sütun) ──
    with sekme_rastgele:
        sol, sag = st.columns([1.45, 1], gap="medium")

        with sol:
            st.markdown("### 🎲 Tehdit tarayıcı")
            if st.button("🔄 Yeni örnek yükle", type="primary", use_container_width=True):
                st.session_state.ornek_i = int(np.random.randint(0, len(X_test)))
                st.session_state.sim_tohum = int(np.random.randint(0, 10_000))

            if "ornek_i" not in st.session_state:
                st.session_state.ornek_i = int(np.random.randint(0, len(X_test)))

            i = st.session_state.ornek_i
            statik = X_test[i : i + 1]
            dizi = X_seq_test[i : i + 1]
            ham = df_test.iloc[int(gecerli_idx[i])]
            proto, servis, bayrak = _ham_satir_alanlari(ham)
            aktif = len(
                aktif_ozellikleri_cikar(statik.ravel(), veri["ozellik_adlari"], veri["sayisal_sutunlar"])
            )

            ornek_ozet_karti(
                ornek_no=int(gecerli_idx[i]),
                gercek_sinif=sinif_adlari[int(y_test[i])],
                protokol=proto,
                servis=servis,
                bayrak=bayrak,
                aktif_ozellik=aktif,
                toplam_ozellik=ozellik_sayisi,
            )

            with st.expander("📊 Aktif özellik sinyalleri", expanded=True):
                ozellik_bolumlerini_goster(
                    statik.ravel(),
                    veri["ozellik_adlari"],
                    veri["sayisal_sutunlar"],
                    toplam_boyut=ozellik_sayisi,
                )

            if st.button("🎯 Tehdit analizi başlat", use_container_width=True, key="tahmin_r"):
                st.markdown("#### 🔮 Analiz sonucu")
                _tahmin_akisi(
                    model, veri, statik, dizi, gercek_sinif=int(y_test[i]), ham_satir=ham
                )

        with sag:
            trafik_izleyici_goster(df_test, veri["y_test_ham"], sinif_adlari)

    # ── Sekme 2: Manuel ──
    with sekme_manuel:
        st.markdown("### 🖊️ Manuel prob")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📥 Test örneği yükle", use_container_width=True):
                ri = int(np.random.randint(0, len(X_test)))
                st.session_state.manuel_ozellikler = X_test[ri].copy()
                st.session_state.manuel_dizi = X_seq_test[ri].copy()
                st.session_state.manuel_gercek = int(y_test[ri])

        if "manuel_ozellikler" not in st.session_state:
            st.session_state.manuel_ozellikler = np.zeros(ozellik_sayisi, dtype=np.float32)
            st.session_state.manuel_dizi = np.zeros((pencere, ozellik_sayisi), dtype=np.float32)

        with st.expander("📊 Aktif özellikler", expanded=True):
            ozellik_bolumlerini_goster(
                st.session_state.manuel_ozellikler,
                veri["ozellik_adlari"],
                veri["sayisal_sutunlar"],
                toplam_boyut=ozellik_sayisi,
            )

        if st.button("🎯 Tahmin et", type="primary", use_container_width=True):
            statik = np.array(st.session_state.manuel_ozellikler, dtype=np.float32).reshape(1, -1)
            dizi = np.array(st.session_state.manuel_dizi, dtype=np.float32).reshape(1, pencere, -1)
            _tahmin_akisi(
                model,
                veri,
                statik,
                dizi,
                gercek_sinif=st.session_state.get("manuel_gercek"),
                ham_satir=None,
            )

    # ── Sekme 3: Model + grafikler ──
    with sekme_model:
        st.markdown("### 📈 Sistem istihbaratı")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{ozellik_sayisi}</div>'
                f'<div class="stat-label">Girdi boyutu</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{pencere}</div>'
                f'<div class="stat-label">LSTM penceresi</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">5</div>'
                f'<div class="stat-label">Çıkış sınıfı</div></div>',
                unsafe_allow_html=True,
            )

        render_analytics_panel(
            veri["y_egitim"],
            y_test,
            st.session_state.tahmin_gecmisi,
            st.session_state.sim_tohum,
            sinif_adlari=sinif_adlari,
            smote_before=veri.get("smote_once"),
            smote_after=veri.get("smote_sonra"),
        )

    st.caption("DNN-IDS · SOC Paneli · Hibrit çok sınıflı IDS")


if __name__ == "__main__":
    main()
