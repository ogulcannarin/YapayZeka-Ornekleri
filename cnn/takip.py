import cv2
from ultralytics import YOLO

# 1. MODELİ YÜKLE
# 'yolov8n.pt' (nano) en hızlı olanıdır, bilgisayarı kasmaz.
# İlk çalıştırdığında bu dosyayı internetten otomatik indirecek.
print("Model yükleniyor...")
model = YOLO('yolov8n.pt') 

# 2. KAMERAYI AÇ (0 genellikle bilgisayarın kendi web kamerasıdır)
cap = cv2.VideoCapture(0)

while True:
    # Kameradan bir kare (fotoğraf) oku
    ret, frame = cap.read()
    if not ret:
        break

    # 3. YAPAY ZEKAYA GÖSTER (Tahmin Et)
    # stream=True, videonun daha akıcı olmasını sağlar
    results = model(frame, stream=True)

    # 4. SONUÇLARI ÇİZ
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Kutunun koordinatlarını al
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Sınıf ismini al (Örn: person, cup, cell phone)
            cls = int(box.cls[0])
            class_name = model.names[cls]

            # SADECE İNSANLARI TAKİP ET (Filtreleme)
            if class_name == 'person':
                # Kutuyu çiz (Renk: Yeşil, Kalınlık: 3)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Yazıyı yaz
                cv2.putText(frame, "Cocuk", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # KONUM BİLGİSİNİ KONSOLA YAZ (Senin istediğin özellik)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                print(f"Hedef Tespit Edildi! Merkez Konum: X={center_x}, Y={center_y}")

    # Görüntüyü ekrana yansıt
    cv2.imshow('Cocuk Takip Sistemi', frame)

    # 'q' tuşuna basınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()