import pandas as pd
import numpy as np
from preprocess_nsl_kdd import attack_name_to_category, CLASS_NAME_TO_ID, multiclass_labels

df_train = pd.read_parquet('data/KDDTrain.parquet')

print("=" * 70)
print("ÖRNEK: VERİ SETİNDEKİ METIN ETİKETLER → SAYISAL ID DÖNÜŞÜMÜ")
print("=" * 70)

# İlk 20 örneğin gerçek etiketlerini göster
print("\n1. VERİ SETİNDE YAZILI OLAN ETİKETLER (Metin):")
print("-" * 70)
sample_labels = df_train['class'].head(20).values
for i, label in enumerate(sample_labels):
    print(f"  Örnek {i+1}: '{label}'")

print("\n2. BU ETİKETLERİ KATEGORILERE DÖNÜŞTÜRME:")
print("-" * 70)
for i, label in enumerate(sample_labels):
    category = attack_name_to_category(label)
    numeric_id = CLASS_NAME_TO_ID[category]
    print(f"  '{label}' → '{category}' → {numeric_id}")

print("\n3. TÜMMÜ VEYA (20 örneğin sonucu):")
print("-" * 70)
numeric_labels = multiclass_labels(df_train.head(20))
print("Sayısal etiketler:", numeric_labels)

print("\n4. SALDIRI TÜRLERI MAPPING TABLOSU:")
print("-" * 70)
print("\nNORMAL → 0")
print("  • 'normal' → 0")
print("\nDoS SALDIRISI → 1")
print("  • 'neptune', 'back', 'smurf', 'teardrop', 'pod', 'land' → 1")
print("  • Yeni (test-only): 'apache2', 'mailbomb', 'processtable', 'udpstorm', 'worm' → 1")
print("\nPROBE (KEŞİF) → 2")
print("  • 'ipsweep', 'nmap', 'portsweep', 'satan' → 2")
print("  • Yeni (test-only): 'mscan', 'saint' → 2")
print("\nR2L (UZAKTAN ERİŞİM SALDIRISI) → 3")
print("  • 'ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy',")
print("    'warezclient', 'warezmaster' → 3")
print("  • Yeni (test-only): 'httptunnel', 'named', 'ps', 'sendmail',")
print("    'snmpgetattack', 'snmpguess', 'sqlattack', 'xlock', 'xsnoop', 'xterm' → 3")
print("\nU2R (YETKİLİ ERIŞIM SALDIRISI) → 4")
print("  • 'buffer_overflow', 'loadmodule', 'perl', 'rootkit' → 4")
