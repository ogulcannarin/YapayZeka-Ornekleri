from transformers import pipeline

print("Model hazırlanıyor...")
analizci = pipeline("sentiment-analysis", model="savasy/bert-base-turkish-sentiment-cased")

ornek_cumleler = [
    "Bu ürünü çok sevdim, harika çalışıyor!",
    "Kargo çok geç geldi ve paket ezilmişti.",
    "Fena değil ama fiyatı biraz pahalı gibi.",
    "Yapay zeka öğrenmek sandığımdan çok daha zevkliymiş.",
    "Bugün hava çok güzel."
]

print("\n--- TEST SONUÇLARI ---\n")

for cumle in ornek_cumleler:
    sonuc = analizci(cumle)[0]
    
    # --- DÜZELTME BURADA ---
    # Model bazen 'positive' bazen 'LABEL_1' diyebilir. İkisini de kapsayalım.
    gelen_etiket = sonuc['label']
    
    if gelen_etiket in ['positive', 'LABEL_1']:
        duygu = "POZİTİF 😊"
    else:
        duygu = "NEGATİF 😡"
        
    guven = sonuc['score'] * 100
    
    print(f"Cümle: {cumle}")
    print(f"Yorum: {duygu} (Eminlik: %{guven:.2f})")
    print(f"Ham Etiket: {gelen_etiket}") # Hata ayıklamak için etiketi de görelim
    print("-" * 30)

while True:
    kullanici_giris = input("\nBir cümle yaz (Çıkış 'q'): ")
    if kullanici_giris.lower() == 'q':
        break
    
    res = analizci(kullanici_giris)[0]
    
    # Aynı düzeltme burada da var
    if res['label'] in ['positive', 'LABEL_1']:
        etiket = "POZİTİF 😊"
    else:
        etiket = "NEGATİF 😡"
        
    print(f">> AI Analizi: {etiket} (Güven: %{res['score']*100:.1f})")