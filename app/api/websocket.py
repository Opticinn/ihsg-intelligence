"""
app/api/websocket.py
====================
Endpoint Real-Time WebSockets.
User yang konek ke sini akan otomatis mendapat notifikasi jika ada sinyal ML baru.
"""

import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.event_bus import redis_client

router = APIRouter(tags=["Real-time Signals"])

@router.websocket("/ws/signals")
async def websocket_endpoint(websocket: WebSocket):
    # 1. Terima koneksi dari browser
    await websocket.accept()
    print("🔌 Klien baru terhubung ke WebSocket!")
    
    # 2. Siapkan 'Radio Penerima' (Redis Pub/Sub)
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("ihsg_signals")
    
    try:
        while True:
            # 3. Dengarkan siaran dari Redis terus-menerus
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            
            if message:
                # 4. Jika ada pesan (misal XGBoost selesai prediksi), teruskan ke Browser!
                data = message["data"]
                await websocket.send_text(data)
            else:
                # Jeda sejenak agar CPU laptopmu tidak meledak ke 100%
                await asyncio.sleep(0.1) 
                
    except WebSocketDisconnect:
        print("🔌 Klien terputus dari WebSocket.")
    except Exception as e:
        print(f"⚠️ Error WebSocket: {e}")
    finally:
        # Bersihkan koneksi saat user menutup browser
        await pubsub.unsubscribe("ihsg_signals")
        await pubsub.close()