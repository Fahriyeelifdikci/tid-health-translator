"""
el_noktalari.py — TİD bitirme projesi, 2 haftalık deneme / 1. adım
------------------------------------------------------------------
Amaç: Bir videodan (veya webcam'den) her karede el noktalarını (21 nokta x 2 el)
çıkarmak, ekranda çizmek ve sonucu .npy dosyasına kaydetmek.

Neden önemli? Projede model ham videoyu değil, bu noktaları görecek.
Yani burada ürettiğin dizi = LSTM/Transformer'ın girdisi.

KURULUM (bir kere):
    python -m venv venv                       # Python 3.11 önerilir
    venv\\Scripts\\activate                    # Windows  (Mac/Linux: source venv/bin/activate)
    pip install mediapipe opencv-python numpy

MODEL DOSYASI (bir kere, bu script ile aynı klasöre indir):
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

ÇALIŞTIRMA:
    python el_noktalari.py                    # webcam (q ile çık)
    python el_noktalari.py ornek_video.mp4    # video dosyası
"""

import sys
import numpy as np
import cv2
import mediapipe as mp

MODEL_YOLU = "hand_landmarker.task"
EL_SAYISI = 2          # TİD'de çoğu işaret iki elle yapılır
NOKTA_SAYISI = 21      # MediaPipe her el için 21 nokta verir (bilek + 4'er parmak eklemi)

# Ekranda noktaları birleştirmek için el iskeleti (hangi nokta hangisine bağlı)
BAGLANTILAR = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # başparmak
    (0, 5), (5, 6), (6, 7), (7, 8),          # işaret parmağı
    (5, 9), (9, 10), (10, 11), (11, 12),     # orta parmak
    (9, 13), (13, 14), (14, 15), (15, 16),   # yüzük parmağı
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # serçe parmak + avuç
]


def landmarker_olustur():
    """HandLandmarker'ı VIDEO modunda kurar (kareler sırayla + zaman damgasıyla gelir)."""
    secenekler = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_YOLU),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=EL_SAYISI,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(secenekler)


def kareyi_vektore_cevir(sonuc):
    """
    Bir karenin sonucunu sabit boyutlu vektöre çevirir: (2 el, 21 nokta, 3 koordinat).
    El görünmüyorsa o kısım 0 kalır. Sabit boyut şart, çünkü model her karede
    aynı uzunlukta girdi bekler.
    Sıra: [0] = Sol el, [1] = Sağ el  (MediaPipe'ın 'handedness' bilgisine göre)
    """
    vektor = np.zeros((EL_SAYISI, NOKTA_SAYISI, 3), dtype=np.float32)
    for el_noktalari, el_bilgisi in zip(sonuc.hand_landmarks, sonuc.handedness):
        taraf = el_bilgisi[0].category_name          # "Left" veya "Right"
        indeks = 0 if taraf == "Left" else 1
        for i, nokta in enumerate(el_noktalari):
            # x, y: görüntüye göre 0-1 arası (normalize); z: bileğe göre derinlik
            vektor[indeks, i] = (nokta.x, nokta.y, nokta.z)
    return vektor


def ciz(kare, sonuc):
    """Noktaları ve iskeleti karenin üstüne çizer."""
    h, w = kare.shape[:2]
    for el_noktalari in sonuc.hand_landmarks:
        pikseller = [(int(n.x * w), int(n.y * h)) for n in el_noktalari]
        for a, b in BAGLANTILAR:
            cv2.line(kare, pikseller[a], pikseller[b], (0, 200, 0), 2)
        for p in pikseller:
            cv2.circle(kare, p, 4, (0, 0, 255), -1)


def main():
    kaynak = sys.argv[1] if len(sys.argv) > 1 else 0   # 0 = webcam
    kamera = cv2.VideoCapture(kaynak)
    if not kamera.isOpened():
        print(f"Kaynak açılamadı: {kaynak}")
        return

    fps = kamera.get(cv2.CAP_PROP_FPS) or 30
    tum_kareler = []        # her karenin (2, 21, 3) vektörü
    el_bulunan = 0
    kare_no = 0

    with landmarker_olustur() as landmarker:
        while True:
            okundu, kare = kamera.read()
            if not okundu:
                break

            # OpenCV BGR verir, MediaPipe RGB ister
            rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
            mp_kare = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # VIDEO modunda milisaniye cinsinden artan zaman damgası şart
            zaman_ms = int(kare_no * 1000 / fps)
            sonuc = landmarker.detect_for_video(mp_kare, zaman_ms)

            tum_kareler.append(kareyi_vektore_cevir(sonuc))
            if sonuc.hand_landmarks:
                el_bulunan += 1

            ciz(kare, sonuc)
            cv2.putText(kare, f"El: {len(sonuc.hand_landmarks)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow("El noktalari (q = cikis)", kare)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            kare_no += 1

    kamera.release()
    cv2.destroyAllWindows()

    if not tum_kareler:
        print("Hiç kare işlenmedi.")
        return

    dizi = np.stack(tum_kareler)            # şekil: (kare_sayisi, 2, 21, 3)
    np.save("el_noktalari.npy", dizi)
    print(f"Kare sayısı      : {len(dizi)}")
    print(f"El bulunan kare  : {el_bulunan} (%{100 * el_bulunan / len(dizi):.1f})")
    print(f"Kaydedilen şekil : {dizi.shape}  ->  el_noktalari.npy")


if __name__ == "__main__":
    main()
