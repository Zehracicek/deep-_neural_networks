# NSL-KDD Tabanlı Derin Öğrenme Saldırı Tespit Sistemi (IDS)

Bu proje, **NSL-KDD** veri kümesi üzerinde ikili sınıflandırma (**normal** / **saldırı**) yapan basit bir **derin sinir ağı (DNN)** ve veri ön işleme hattını içerir. TensorFlow/Keras kullanılır.

**GitHub:** [deep-_neural_networks](https://github.com/Zehracicek/deep-_neural_networks)

## Özellikler

- Parquet formatında NSL-KDD eğitim ve test verisinin yüklenmesi
- Keşifsel veri analizi (EDA)
- Kategorik özelliklerin one-hot kodlanması, sayısal özelliklerin `StandardScaler` ile ölçeklenmesi
- Etiketlerin ikili forma dönüştürülmesi (normal → 0, saldırı → 1)
- Keras ile tam bağlantılı ağ: gizli katmanlar (Dense + ReLU), çıkışta sigmoid
- Sınıf dengesizliği için `class_weight`, **Dropout** ve basit hiperparametre araması
- Test kümesinde doğruluk, kesinlik, duyarlılık, F1 ve karışıklık matrisi

## Gereksinimler

- Python 3.10+ (3.12 ile test edildi)
- Paketler: `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `tensorflow`, `matplotlib`

Kurulum örneği:

```bash
pip install pandas numpy pyarrow scikit-learn tensorflow matplotlib
```

## Proje yapısı

| Dosya | Açıklama |
|--------|-----------|
| `data/KDDTrain.parquet` | Eğitim verisi |
| `data/KDDTest.parquet` | Test verisi |
| `load_nsl_kdd.py` | Parquet okuma, sütun adları, ilk satırlar ve boyut bilgisi |
| `eda_nsl_kdd.py` | Sınıf dağılımı, eksik değer, sayısal/kategorik ayrımı, temel istatistikler |
| `preprocess_nsl_kdd.py` | Ölçekleme, one-hot, ikili etiket, `X_train/y_train`, `X_test/y_test` |
| `dnn_model.py` | Model mimarisi, eğitim (validation split, EarlyStopping), eğri grafikleri |
| `evaluate_dnn.py` | Test tahmini ve metrikler; `confusion_matrix.png` üretimi |
| `improve_ids.py` | Sınıf ağırlığı + Dropout + hiperparametre karşılaştırması (önce/sonra) |
| `training_history.png` | Eğitim/doğrulama kaybı ve doğruluk grafikleri (örnek çıktı) |
| `confusion_matrix.png` | Test karışıklık matrisi (örnek çıktı) |

## Veri kümesi

Bu repodaki Parquet dosyaları **38 sütunluk** bir NSL-KDD türevini kullanır: `protocol_type`, `service`, `flag` kategorik; hedef olarak `class` (metin) ve `classnum` bulunur. İkili sınıflandırmada `normal` dışındaki tüm sınıflar **saldırı (1)** sayılır.

## Kullanım

Proje kök dizininde çalıştırın:

```bash
# Veri yükleme ve ön kontrol
python load_nsl_kdd.py

# EDA
python eda_nsl_kdd.py

# Ön işleme (doğrudan veya diğer betikler içinden)
python -c "from preprocess_nsl_kdd import get_preprocessed_train_test; print(get_preprocessed_train_test()[0].shape)"

# Model özeti ve eğitim + eğitim grafikleri
python dnn_model.py

# Test değerlendirmesi
python evaluate_dnn.py

# İyileştirilmiş model vs taban çizgisi karşılaştırması
python improve_ids.py
```

## Model özeti

- **Girdi:** ön işlemeden sonra birleşik özellik vektörü (ör. ~117 boyut: ölçeklenmiş sayısal + one-hot).
- **Gizli katmanlar:** 3 × Dense + ReLU; iyileştirilmiş sürümde katmanlar arası **Dropout**.
- **Çıkış:** Dense(1) + sigmoid, `binary_crossentropy`, optimizer **Adam**, metrik **accuracy**.
- Eğitimde `validation_split=0.2` ve **EarlyStopping** (`val_loss` izlenir).

## IDS bağlamında duyarlılık (recall)

Saldırı sınıfında **duyarlılık**, gerçek saldırıların ne kadarının yakalandığını gösterir. Yanlış negatif (kaçan saldırı) doğrudan risk oluşturduğu için birçok IDS senaryosunda duyarlılık ve F1, yalnızca doğruluktan daha anlamlı metriklerdir.

## Lisans ve atıf

NSL-KDD veri kümesi üzerine çalışıyorsanız ilgili makaleleri ve kullanım koşullarını kaynakça olarak belirtin. Bu kod öğrenim amaçlı bir örnektir.
