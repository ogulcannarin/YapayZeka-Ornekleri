import cv2
import face_recognition
import numpy as np

# --- HAZIRLIK AŞAMASI (Öğrenme) ---
print("Referans fotoğraf yükleniyor ve işleniyor...")

# 1. Kendi fotoğrafını yükle (ben.jpg dosyasının kodla aynı yerde olduğundan emin ol)
my_image = face_recognition.load_image_file("ben.jpg")

# 2. Fotoğraftaki yüzün matematiksel haritasını (encoding) çıkar
# Bu, yüzü 128 sayılık benzersiz bir listeye dönüştürür.
try:
    my_face_encoding = face_recognition.face_encodings(my_image)[0]
    print("Yüz başarıyla öğrenildi!")
except IndexError:
    print("HATA: ben.jpg fotoğrafında yüz bulunamadı. Lütfen başka bir fotoğrafla deneyin.")
    exit()

# Tanıdığımız yüzlerin listesi
known_face_encodings = [my_face_encoding]
# Bu yüzlerin isimleri
known_face_names = ["BEN"] # Buraya kendi ismini yazabilirsin

# --- KAMERA AŞAMASI ---
video_capture = cv2.VideoCapture(0)

print("Kamera başlatılıyor. Çıkmak için 'q' tuşuna basın.")

while True:
    # 1. Kameradan bir kare al
    ret, frame = video_capture.read()
    if not ret:
        break

    # 2. İşlemi hızlandırmak için görüntüyü küçült (1/4 oranında)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # 3. OpenCV BGR renk kullanır, face_recognition RGB ister. Dönüşüm yapıyoruz.
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

    # 4. Küçültülmüş karedeki tüm yüzlerin yerlerini ve haritalarını bul
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # 5. Bulunan yüzü, tanıdığımız yüzlerle karşılaştır
        # tolerance=0.6 standarttır. Daha düşük yaparsan (örn 0.5) daha katı olur, zor tanır.
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Yabanci"

        # Eğer eşleşme varsa ilk bulduğu ismi kullan
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    # 6. Sonuçları ekrana çiz (Koordinatları tekrar büyütmemiz lazım çünkü küçültmüştük)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Kutu rengi: Tanıdıysa Yeşil, Tanımadıysa Kırmızı
        box_color = (0, 255, 0) if name != "Yabanci" else (0, 0, 255)

        # Yüzün etrafına kutu çiz
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)

        # İsmi yazmak için kutunun altına bir etiket alanı çiz
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Görüntüyü göster
    cv2.imshow('Yuz Tanima Sistemi', frame)

    # 'q' tuşuna basınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()