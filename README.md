# 🧠 Yapay Zeka Örnekleri

Bu repo, **Derin Öğrenme** ve **Yapay Zeka** alanındaki farklı sinir ağı mimarilerini öğrenmek isteyenler için hazırlanmış **Türkçe** açıklamalı örnek projeler içermektedir. Her proje, gerçek dünya uygulamalarıyla desteklenmiş ve başlangıç seviyesindeki kullanıcılar için anlaşılır hale getirilmiştir.

---

## 📚 İçindekiler

- [Proje Yapısı](#-proje-yapısı)
- [Projeler](#-projeler)
  - [1. CNN (Convolutional Neural Networks)](#1-cnn---evrişimli-sinir-ağları)
  - [2. RNN (Recurrent Neural Networks)](#2-rnn---tekrarlayan-sinir-ağları)
  - [3. GNN (Graph Neural Networks)](#3-gnn---graf-sinir-ağları)
  - [4. Transformer](#4-transformer---dikkat-mekanizması)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Gereksinimler](#-gereksinimler)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Lisans](#-lisans)

---

## 📁 Proje Yapısı

```
YapayZeka-Ornekleri/
│
├── cnn/                    # Görüntü işleme projeleri
│   ├── takip.py           # YOLOv8 ile nesne takibi
│   ├── yuz_tanima.py      # Yüz tanıma sistemi
│   ├── ben.jpg            # Referans görsel
│   ├── yolov8n.pt         # YOLOv8 model ağırlıkları
│   └── requirements.txt
│
├── rnn/                    # Dizi tabanlı projeler
│   ├── rnn_yazar.py       # Metin üreten RNN
│   └── requirements.txt
│
├── gnn/                    # Graf tabanlı projeler
│   ├── gnn_ornek.py       # Karate Club sınıflandırma
│   └── requirements.txt
│
└── transformer/            # Dikkat mekanizması projeleri
    ├── transformer_ornek.py  # Duygu analizi
    └── requirements.txt
```

---

## 🚀 Projeler

### 1. CNN - Evrişimli Sinir Ağları

Görüntü işleme ve bilgisayarlı görü uygulamaları için CNN mimarisi kullanılmıştır.

#### 🎯 **Nesne Takip Sistemi** (`takip.py`)
- **Amaç**: Kamera görüntüsünden gerçek zamanlı insan tespiti ve takibi
- **Teknoloji**: YOLOv8 (ultralytics)
- **Özellikler**:
  - Webcam üzerinden canlı video akışı
  - İnsan tespiti (person detection)
  - Nesne konumu (x, y koordinatları) hesaplama
  - Yeşil kutu ile görsel işaretleme

**Kullanım:**
```bash
cd cnn
python takip.py
# Çıkmak için 'q' tuşuna basın
```

#### 👤 **Yüz Tanıma Sistemi** (`yuz_tanima.py`)
- **Amaç**: Referans fotoğrafla kamera görüntüsündeki yüzleri karşılaştırma
- **Teknoloji**: face_recognition + OpenCV
- **Özellikler**:
  - 128 boyutlu yüz kodlaması (face encoding)
  - Gerçek zamanlı yüz eşleştirme
  - Tanınan yüzler yeşil, yabancılar kırmızı kutu ile işaretlenir
  - Performans optimizasyonu (1/4 görüntü ölçeklendirme)

**Kullanım:**
```bash
cd cnn
# ben.jpg dosyasını kendi fotoğrafınızla değiştirin
python yuz_tanima.py
```

---

### 2. RNN - Tekrarlayan Sinir Ağları

Sıralı veri işleme ve doğal dil işleme (NLP) uygulamaları için RNN mimarisi.

#### ✍️ **Metin Üretici RNN** (`rnn_yazar.py`)
- **Amaç**: Karakter bazlı dil modeli ile otomatik metin üretimi
- **Teknoloji**: PyTorch RNN
- **Özellikler**:
  - Embedding katmanı ile karakter vektörleştirme
  - Sıralı veri öğrenme (sequence learning)
  - Greedy decoding ile metin tamamlama
  - 100 epoch eğitim döngüsü

**Nasıl Çalışır?**
1. Model, verilen metindeki karakter dizilerini öğrenir
2. "yapay" gibi bir başlangıç kelimesi verilir
3. Model, sonraki karakterleri tahmin ederek metni tamamlar

**Kullanım:**
```bash
cd rnn
python rnn_yazar.py
```

**Örnek Çıktı:**
```
yapay zeka python ile kodlama yapmak cok eglenceli...
```

---

### 3. GNN - Graf Sinir Ağları

İlişkisel veri yapıları üzerinde öğrenme yapan graf tabanlı modellerdir.

#### 🥋 **Karate Club Sınıflandırma** (`gnn_ornek.py`)
- **Amaç**: Sosyal ağ analizi ve topluluk tespiti
- **Teknoloji**: PyTorch Geometric (GCN)
- **Dataset**: Zachary's Karate Club
- **Özellikler**:
  - 2 katmanlı GCN (Graph Convolutional Network)
  - 34 düğüm (kulüp üyeleri) üzerinde grup tahmini
  - NetworkX ile görselleştirme
  - Semi-supervised learning (yarı gözetimli öğrenme)

**Kullanım:**
```bash
cd gnn
python gnn_ornek.py
```

**Görsel Çıktı:**  
Model, kulüp üyelerini iki gruba ayırarak renkli bir graf gösterir.

---

### 4. Transformer - Dikkat Mekanizması

Self-Attention mekanizması ile doğal dil işleme ve metin analizi için modern transformer mimarisi.

#### 💬 **Türkçe Duygu Analizi** (`transformer_ornek.py`)
- **Amaç**: Türkçe metinlerde duygu (sentiment) analizi yapma
- **Teknoloji**: Hugging Face Transformers (BERT)
- **Model**: `savasy/bert-base-turkish-sentiment-cased`
- **Özellikler**:
  - Türkçe'ye özel eğitilmiş BERT modeli
  - Pozitif/Negatif duygu sınıflandırması
  - Güven skoru hesaplama (confidence score)
  - İnteraktif test modu
  - Hazır örnek cümlelerle demo

**Nasıl Çalışır?**
1. Pre-trained Türkçe BERT modeli yüklenir
2. Verilen cümle, model tarafından analiz edilir
3. Cümlenin pozitif/negatif olma olasılığı hesaplanır
4. Sonuç emoji ile birlikte gösterilir 😊/😡

**Kullanım:**
```bash
cd transformer
pip install -r requirements.txt
python transformer_ornek.py
```

**Örnek Çıktı:**
```
Cümle: Bu ürünü çok sevdim, harika çalışıyor!
Yorum: POZİTİF 😊 (Eminlik: %99.87)

Cümle: Kargo çok geç geldi ve paket ezilmişti.
Yorum: NEGATİF 😡 (Eminlik: %98.45)
```

**Kullanım Alanları:**
- Sosyal medya analizi
- Müşteri yorumu izleme
- Ürün inceleme değerlendirme
- Chatbot duygu tespiti

---

## 🛠 Kurulum

### 1. Repoyu klonlayın
```bash
git clone https://github.com/ogulcannarin/YapayZeka-Ornekleri.git
cd YapayZeka-Ornekleri
```

### 2. Sanal ortam oluşturun (önerilen)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Bağımlılıkları yükleyin
Her projenin kendi `requirements.txt` dosyası vardır:

```bash
# CNN için
cd cnn
pip install -r requirements.txt

# RNN için
cd rnn
pip install -r requirements.txt

# GNN için
cd gnn
pip install -r requirements.txt

# Transformer için
cd transformer
pip install -r requirements.txt
```

---

## 💻 Gereksinimler

### Genel Gereksinimler
- Python 3.8+
- pip paket yöneticisi
- Webcam (CNN projeleri için)

### Proje Bazlı Kütüphaneler

**CNN:**
- opencv-python
- ultralytics (YOLOv8)
- face_recognition
- numpy

**RNN:**
- torch
- numpy

**GNN:**
- torch
- torch-geometric
- networkx
- matplotlib

**Transformer:**
- transformers
- torch
- sentencepiece

> **Not:** YOLOv8 ilk çalıştırmada model ağırlıklarını otomatik olarak indirecektir (~6MB).

---

## 📖 Kullanım

Her proje bağımsız çalışabilir. İlgilendiğiniz klasöre gidip ilgili Python dosyasını çalıştırmanız yeterlidir:

```bash
# Örnek: Yüz tanıma projesini çalıştırma
cd cnn
python yuz_tanima.py
```

**Dikkat Edilmesi Gerekenler:**
- `yuz_tanima.py` için klasörde `ben.jpg` dosyası olmalıdır
- Kamera izinlerinin verilmiş olması gerekir
- İlk çalıştırmalarda model indirmeleri için internet bağlantısı gereklidir

---

## 🎓 Öğrenme Kaynakları

Bu projeler aşağıdaki kavramları öğrenmek için harika bir başlangıç noktasıdır:

- **CNN**: Görüntü işleme, nesne tespiti, yüz tanıma
- **RNN**: Zaman serisi analizi, metin üretimi, doğal dil işleme
- **GNN**: Graf analizi, sosyal ağ madenciliği, molekül sınıflandırma
- **Transformer**: Self-attention mekanizması, BERT modeli, duygu analizi, modern NLP

---

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Katkıda bulunmak için:

1. Bu repoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin yeni-ozellik`)
5. Pull Request açın

---

## 📝 Lisans

Bu proje MIT lisansı altında sunulmaktadır. Detaylar için `LICENSE` dosyasına bakabilirsiniz.

---

## 📧 İletişim

Sorularınız veya önerileriniz için:
- GitHub: [@ogulcannarin](https://github.com/ogulcannarin)
- Issues: [Sorun Bildir](https://github.com/ogulcannarin/YapayZeka-Ornekleri/issues)

---

## ⭐ Destek

Bu projeyi faydalı bulduysanız, yıldız ⭐ vermeyi unutmayın!

---

**Not:** Bu projeler eğitim amaçlıdır ve üretim ortamlarında kullanılmadan önce optimize edilmelidir.
