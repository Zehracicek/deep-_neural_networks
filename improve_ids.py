"""
Compare baseline vs improved hybrid multiclass IDS (class weights + dropout tuning).

Uses 5-class NSL-KDD labels, SMOTE on training data, and macro-F1 on validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils as keras_utils

from evaluate_dnn import compute_multiclass_metrics
from hybrid_model import multiclass_class_weights, train_hybrid_model
from preprocess_nsl_kdd import class_distribution, prepare_hybrid_dataset
from smote_balance import print_smote_report


@dataclass(frozen=True)
class TuneConfig:
    etiket: str
    dropout: float
    learning_rate: float
    dnn_units: tuple[int, int, int]


def en_iyi_val_loss(gecmis) -> float:
    return float(min(gecmis.history["val_loss"]))


def main() -> None:
    print("Hibrit veri (SMOTE + pencere) yükleniyor…")
    veri = prepare_hybrid_dataset(apply_smote=True)
    if veri.smote_before:
        print_smote_report(veri.smote_before, veri.smote_after)

    agirliklar = multiclass_class_weights(veri.y_train)
    print("Dengeli sınıf ağırlıkları:", agirliklar)
    print("Eğitim dağılımı:", class_distribution(veri.y_train))
    print()

    X_tr, X_val, X_seq_tr, X_seq_val, y_tr, y_val = train_test_split(
        veri.X_train,
        veri.X_seq_train,
        veri.y_train,
        test_size=0.2,
        stratify=veri.y_train,
        random_state=42,
    )

    print("=== Taban çizgi (düşük dropout) ===")
    keras_utils.set_random_seed(42)
    taban, taban_gecmis = train_hybrid_model(
        X_tr,
        X_seq_tr,
        y_tr,
        epochs=12,
        validation_data=(X_val, X_seq_val, y_val),
        verbose=0,
        class_weight=agirliklar,
        dropout=0.15,
        window_size=veri.window_size,
    )
    taban_met = compute_multiclass_metrics(taban, veri.X_test, veri.X_seq_test, veri.y_test)
    print(
        f"Test F1(macro)={taban_met['f1_macro']:.4f}  "
        f"ROC-AUC={taban_met['roc_auc_ovr_macro']:.4f}  "
        f"val_loss={en_iyi_val_loss(taban_gecmis):.4f}"
    )
    print()

    adaylar = [
        TuneConfig("d0.25_lr1e-3", 0.25, 1e-3, (128, 64, 32)),
        TuneConfig("d0.35_lr5e-4", 0.35, 5e-4, (128, 64, 32)),
        TuneConfig("d0.2_h256", 0.2, 1e-3, (256, 128, 64)),
    ]
    en_iyi: TuneConfig | None = None
    en_iyi_skor = (-1.0, float("inf"))

    print("=== Hiperparametre araması (makro F1) ===")
    for i, cfg in enumerate(adaylar):
        keras_utils.set_random_seed(100 + i)
        model, gecmis = train_hybrid_model(
            X_tr,
            X_seq_tr,
            y_tr,
            epochs=12,
            validation_data=(X_val, X_seq_val, y_val),
            verbose=0,
            class_weight=agirliklar,
            window_size=veri.window_size,
            learning_rate=cfg.learning_rate,
            dnn_units=cfg.dnn_units,
            dropout=cfg.dropout,
        )
        met = compute_multiclass_metrics(model, X_val, X_seq_val, y_val)
        skor = (met["f1_macro"], en_iyi_val_loss(gecmis))
        print(f"  [{cfg.etiket}] val F1={met['f1_macro']:.4f}  val_loss={skor[1]:.4f}")
        if skor[0] > en_iyi_skor[0] or (skor[0] == en_iyi_skor[0] and skor[1] < en_iyi_skor[1]):
            en_iyi_skor = skor
            en_iyi = cfg

    assert en_iyi is not None
    print(f"\nSeçilen: {en_iyi.etiket}\n")

    print("=== İyileştirilmiş model (tam eğitim dilimi) ===")
    keras_utils.set_random_seed(999)
    iyilestirilmis, _ = train_hybrid_model(
        X_tr,
        X_seq_tr,
        y_tr,
        epochs=12,
        validation_data=(X_val, X_seq_val, y_val),
        verbose=0,
        class_weight=agirliklar,
        dnn_units=en_iyi.dnn_units,
        dropout=en_iyi.dropout,
        learning_rate=en_iyi.learning_rate,
        window_size=veri.window_size,
    )  # dnn_units/dropout via build inside train_hybrid_model
    imp_met = compute_multiclass_metrics(
        iyilestirilmis, veri.X_test, veri.X_seq_test, veri.y_test
    )
    print(
        f"Test F1(macro)={imp_met['f1_macro']:.4f}  "
        f"ROC-AUC={imp_met['roc_auc_ovr_macro']:.4f}"
    )
    print(f"Δ F1 = {imp_met['f1_macro'] - taban_met['f1_macro']:+.4f}")


if __name__ == "__main__":
    main()
