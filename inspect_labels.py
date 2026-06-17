import pandas as pd

df_train = pd.read_parquet('data/KDDTrain.parquet')
df_test = pd.read_parquet('data/KDDTest.parquet')

print('=== EĞİTİM VERİSİNDE BULUNAN SALDIRI TÜRLERI ===')
print(df_train['class'].value_counts())
print(f'\nToplam eşsiz saldırı türü: {df_train["class"].nunique()}')

print('\n=== TEST VERİSİNDE BULUNAN SALDIRI TÜRLERI ===')
print(df_test['class'].value_counts())
print(f'\nToplam eşsiz saldırı türü: {df_test["class"].nunique()}')

print('\n=== TEST VERİSİNDE SADECE OLAN SALDIRI TÜRLERI (EĞİTİMDE YOK) ===')
train_classes = set(df_train['class'].unique())
test_classes = set(df_test['class'].unique())
only_in_test = test_classes - train_classes
print(list(only_in_test))

print('\n=== ORİJİNAL SALDIRI ADLARI → KATEGORILER MAPPING ===')
print('DoS Saldırıları: back, land, neptune, pod, smurf, teardrop')
print('Probe (Tarama): ipsweep, nmap, portsweep, satan')
print('R2L (Uzaktan Erişim): ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster')
print('U2R (Yetkili Erişim): buffer_overflow, loadmodule, perl, rootkit')
print('\nFinal Kategori Mapping:')
print('Normal (0), DoS (1), Probe (2), R2L (3), U2R (4)')
