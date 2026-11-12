# Sign to Text: AI-Powered Turkish Sign Language Translator for Healthcare

> **TR:** Sağlık ortamında işitme engelli hastaların Türk İşaret Dili (TİD) işaretlerini kameradan tanıyıp doktorun okuyabileceği doğal Türkçe cümlelere çeviren bir sistem. Balıkesir Üniversitesi Bilgisayar Mühendisliği bitirme projesi.

Deaf patients often struggle to communicate with doctors and pharmacists, because most healthcare workers do not know Turkish Sign Language and interpreters are not always available. This project recognizes Turkish Sign Language signs from video and turns them into natural Turkish sentences.

**Status:** 🚧 In progress (graduation project, Nov 2026 – Jun 2027)

## How it works

```
Video → MediaPipe (hand & body keypoints) → LSTM / Transformer (sign recognition) → LLM (natural Turkish sentence) → Web UI
```

Example: `BEN BAŞ AĞRI İKİ GÜN` → *"İki gündür başım ağrıyor."*

## Tech stack

| Layer | Technology |
|---|---|
| Keypoint extraction | MediaPipe, OpenCV |
| Model training | Python, PyTorch, scikit-learn |
| Model service | FastAPI |
| Sentence generation | LLM |
| Web application | ASP.NET Core MVC, Entity Framework Core, SQL Server |

## Roadmap

- [x] Extract hand keypoints from webcam/video with MediaPipe
- [ ] Prepare dataset (BosphorusSign22k / AUTSL)
- [ ] Baseline model (Random Forest)
- [ ] LSTM and Transformer models
- [ ] LLM sentence generation
- [ ] ASP.NET Core web interface
- [ ] AI agent for patient notes
- [ ] Speech-to-text for two-way communication (Whisper)

## Quick start

```bash
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt   # Windows
```

Download the [hand landmarker model](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) into the project folder, then run:

```bash
venv\Scripts\python el_noktalari.py              # webcam
venv\Scripts\python el_noktalari.py video.mp4    # video file
```

Press `q` to quit. Keypoints are saved to `el_noktalari.npy` with shape `(frames, 2 hands, 21 points, 3 coords)`.

## Disclaimer

This system is a communication aid only. It is **not** a medical diagnosis tool.

## Author

**Fahriye Elif Dikci** · Computer Engineering, Balıkesir University
Advisor: Dr. Fatih Aydın
