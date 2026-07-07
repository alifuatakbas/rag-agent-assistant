#!/bin/bash
# ===== Not Asistanı başlatıcı =====

echo "🧠 Not Asistanı başlatılıyor..."

# Backend'i arka planda başlat
cd "$(dirname "$0")/backend"
source .venv/bin/activate
uvicorn api:app --port 4800 &
BACKEND_PID=$!
echo "✓ Backend başladı (arka planda)"

# Backend'in hazır olması için birkaç saniye bekle
sleep 3

# Frontend'i başlat
cd ../frontend
echo "✓ Frontend başlıyor... Tarayıcıda http://localhost:3000 açılacak"
npm run dev

# Frontend kapanınca backend'i de kapat
kill $BACKEND_PID
echo "Not Asistanı kapatıldı."