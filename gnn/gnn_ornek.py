import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# 1. Verisetini Yükle (Karate Kulübü)
dataset = KarateClub()
data = dataset[0]  # Tek bir grafımız var

print(f"Düğüm sayısı (Öğrenciler): {data.num_nodes}")
print(f"Kenar sayısı (Bağlantılar): {data.num_edges}")
print(f"Sınıf sayısı (Gruplar): {dataset.num_classes}")

# 2. GNN Modelini Tanımla (GCN - Graph Convolutional Network)
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(42)
        # İlk katman: Girdiden 4 özelliğe indir
        self.conv1 = GCNConv(dataset.num_features, 4)
        # İkinci katman: 4 özellikten sonuç sınıflarına (gruplara) indir
        self.conv2 = GCNConv(4, dataset.num_classes)

    def forward(self, x, edge_index):
        # x: Düğüm özellikleri, edge_index: Bağlantı bilgisi
        h = self.conv1(x, edge_index)
        h = h.tanh()  # Aktivasyon fonksiyonu
        h = self.conv2(h, edge_index)
        return h, F.log_softmax(h, dim=1)

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# 3. Eğitimi Başlat
def train():
    model.train()
    optimizer.zero_grad()
    # Modeli çalıştır
    _, out = model(data.x, data.edge_index)
    # Sadece eğitim maskesi olan düğümlerle (etiketli olanlar) kaybı hesapla
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

# 4. Görselleştirme Fonksiyonu
def visualize(h, color):
    z = torch.argmax(h, dim=1)
    
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    # NetworkX kullanarak çiz
    G = nx.Graph()
    edge_index = data.edge_index.numpy()
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=color, cmap="Set2")
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")
    
    plt.title("Karate Kulübü Grupları (GNN Tahmini)")
    plt.show()

# Eğitimi döngüye sok
print("\nEğitim Başlıyor...")
for epoch in range(201):
    loss = train()
    if epoch % 20 == 0:
        print(f'Epoch: {epoch}, Loss: {loss:.4f}')

# Sonuçları görselleştir
model.eval()
_, out = model(data.x, data.edge_index)
visualize(out, color=out.argmax(dim=1))