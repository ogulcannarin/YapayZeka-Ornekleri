import torch
import torch.nn as nn
import numpy as np

# --- 1. VERİ HAZIRLIĞI ---
# Yapay zekanın öğrenmesini istediğimiz metin.
# (Kısa tuttum ki işlemcin hızlıca eğitebilsin ve sonucu hemen gör)
text = "yapay zeka python ile kodlama yapmak cok eglenceli bir is. " * 50

# Harfleri sayıya çevirmemiz lazım (Bilgisayar harften anlamaz)
chars = list(set(text))  # Benzersiz karakterleri bul
indexer = {char: i for i, char in enumerate(chars)} # a->0, b->1 gibi sözlük oluştur
decoder = {i: char for i, char in enumerate(chars)} # 0->a, 1->b geri dönüşüm

# Veriyi sayısal diziye dök
data = [indexer[c] for c in text]

# Girdi (X) ve Hedef (Y) oluşturma
# Örnek: "yapa" -> "apay" (Bir sonraki harfi tahmin et)
def get_batch(data, seq_len):
    inputs = []
    targets = []
    for i in range(len(data) - seq_len):
        inputs.append(data[i:i+seq_len])
        targets.append(data[i+1:i+seq_len+1])
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

# --- 2. RNN MODELİ ---
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # Harfleri vektöre çeviren katman (Embedding)
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # RNN Katmanı: İşte projenin kalbi burası
        # batch_first=True: Veri (Batch, Seq, Feature) sırasında gelir
        self.rnn = nn.RNN(hidden_size, hidden_size, n_layers, batch_first=True)
        
        # Çıktı katmanı: Hangi harf olma ihtimali yüksek?
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        # x: Girdi harfleri
        # hidden: Önceki harften kalan hafıza
        
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        
        # Çıktıyı düzleştir ve tahmin üret
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        # İlk başta hafıza sıfırdır
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)

# --- AYARLAR ---
seq_len = 10    # Her seferinde 10 harfe bakıp 11.yi tahmin edecek
hidden_size = 128
lr = 0.005
epochs = 100

model = SimpleRNN(len(chars), hidden_size, len(chars))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

input_seq, target_seq = get_batch(data, seq_len)

# --- 3. EĞİTİM ---
print("Eğitim başlıyor... (Yapay zeka yazmayı öğreniyor)")

for epoch in range(epochs):
    hidden = model.init_hidden(input_seq.size(0))
    optimizer.zero_grad()
    
    output, hidden = model(input_seq, hidden)
    loss = criterion(output, target_seq.view(-1))
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

# --- 4. TEST (YAZDIRMA) ---
print("\n--- Yapay Zekanın Yazdığı Metin ---")

def predict(model, start_text, length):
    model.eval()
    chars_input = [indexer[c] for c in start_text]
    input_seq = torch.tensor(chars_input).unsqueeze(0) # (1, seq_len)
    hidden = model.init_hidden(1)
    
    generated_text = start_text
    
    for _ in range(length):
        output, hidden = model(input_seq, hidden)
        
        # En yüksek ihtimalli harfi seç (Greedy)
        # Sadece son tahmin edilen harfi alıyoruz
        prob = torch.nn.functional.softmax(output, dim=1)
        # Son zaman adımının çıktısını al (-1)
        top_char = prob[-1].argmax().item() # En yüksek olasılıklı harf
        
        char = decoder[top_char]
        generated_text += char
        
        # Yeni girdiyi güncelle (kayan pencere gibi)
        input_seq = torch.cat((input_seq[:, 1:], torch.tensor([[top_char]])), dim=1)

    return generated_text

# Modele "yapay" kelimesini verelim, gerisini o tamamlasın
print(predict(model, "yapay", 50))