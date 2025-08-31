import aiohttp
import aiohttp.web
import asyncio
import os
import json
import glob
from datetime import datetime
import shutil
import cv2
import numpy as np

import stitching_logic

# --- Configuration ---
ESP_IPS = {
    '1': '192.168.137.57',
    # '1': '192.168.137.152',
    '2': '192.168.137.193',
    '3': '192.168.137.129'
}
ESP_WS_URL = "ws://{ip}/ws"
CALIBRATION_RUN_DIR = "data/calibration_run"
ANCHOR_FRAMES_DIR = "data/anchor_frames"

# --- Global State ---
connected_clients = set()
esp_websockets = {}
shared_state = {
    "is_live_calibrating": False,
    "is_capturing_anchors": False,
    "live_stitchers": {},
    "anchor_capture_event": None,
    "anchor_frames_received": {},
}

# --- WebSocket and Networking ---
async def connect_to_esps():
    for esp_id, ip in ESP_IPS.items():
        try:
            session = aiohttp.ClientSession()
            ws = await asyncio.wait_for(session.ws_connect(ESP_WS_URL.format(ip=ip)), timeout=5.0)
            esp_websockets[esp_id] = (session, ws)
            print(f"Successfully connected to ESP {esp_id}")
            # Start listening for messages from this ESP
            asyncio.create_task(listen_to_esp(esp_id, ws))
        except Exception as e:
            print(f"Failed to connect to ESP {esp_id}: {e}")

async def listen_to_esp(esp_id, ws):
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                await handle_esp_binary_message(esp_id, msg.data)
    except Exception as e:
        print(f"Error listening to ESP {esp_id}: {e}")
    finally:
        print(f"ESP {esp_id} disconnected.")
        # Optionally, try to reconnect
        await asyncio.sleep(1)
        asyncio.create_task(connect_to_esps()) # This could be more robust

async def handle_esp_binary_message(esp_id, image_data):
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            print(f"ESP {esp_id}: Failed to decode JPEG - corrupt data (size: {len(image_data)})")
            return

        if not stitching_logic.is_frame_sane(frame):
            print(f"ESP {esp_id}: Frame failed sanity check")
            return

        print(f"ESP {esp_id}: Processing valid frame (shape: {frame.shape})")

        if shared_state["is_live_calibrating"]:
            await handle_live_calibration_frame(esp_id, frame)
        elif shared_state["is_capturing_anchors"]:
            await handle_anchor_capture_frame(esp_id, frame)
        else:
            await handle_normal_frame(esp_id, image_data)

    except Exception as e:
        print(f"ESP {esp_id}: Error processing image frame: {e}")

async def send_to_esp(esp_id, message):
    if esp_id in esp_websockets and not esp_websockets[esp_id][1].closed:
        await esp_websockets[esp_id][1].send_str(message)

async def broadcast_to_web_clients(message):
    for ws in list(connected_clients):
        try:
            await ws.send_json(message)
        except Exception as e:
            print(f"Error broadcasting to client: {e}")
            connected_clients.discard(ws)

async def handle_live_calibration_frame(esp_id, frame):
    stitcher = shared_state["live_stitchers"].get(esp_id)
    if stitcher:
        try:
            timestamp = datetime.now().timestamp()
            frame_path = f"{CALIBRATION_RUN_DIR}/esp_{esp_id}_frame_{timestamp}.jpg"
            cv2.imwrite(frame_path, frame)
            print(f"Saved calibration frame: {frame_path}")
            
            # pano = stitcher.add_frame(frame)
            # if pano is not None:
            #     _, img_encoded = cv2.imencode('.jpg', pano, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            #     message_data = bytes(esp_id, 'utf-8') + img_encoded.tobytes()
                
            #     for ws in list(connected_clients):
            #         try:
            #             await ws.send_bytes(message_data)
            #         except Exception as e:
            #             print(f"Error sending panorama to client: {e}")
            #             connected_clients.discard(ws)
        except Exception as e:
            print(f"ESP {esp_id}: Error in live calibration processing: {e}")

async def handle_anchor_capture_frame(esp_id, frame):
    try:
        filename = f"{ANCHOR_FRAMES_DIR}/anchor_{esp_id}.jpg"
        success = cv2.imwrite(filename, frame)
        if success:
            print(f"Saved anchor frame for ESP {esp_id}: {filename}")
            shared_state["anchor_frames_received"][esp_id] = filename
            
            if len(shared_state["anchor_frames_received"]) == len(ESP_IPS):
                if shared_state["anchor_capture_event"]:
                    shared_state["anchor_capture_event"].set()
        else:
            print(f"Failed to save anchor frame for ESP {esp_id}")
    except Exception as e:
        print(f"ESP {esp_id}: Error saving anchor frame: {e}")

async def handle_normal_frame(esp_id, image_data):
    try:
        message_data = bytes(esp_id, 'utf-8') + image_data
        for ws in list(connected_clients):
            try:
                await ws.send_bytes(message_data)
            except Exception as e:
                print(f"Error sending frame to client: {e}")
                connected_clients.discard(ws)
    except Exception as e:
        print(f"ESP {esp_id}: Error in normal frame handling: {e}")

# --- Web Client Command Handlers & Workflow Logic ---
async def client_handler(request):
    ws = aiohttp.web.WebSocketResponse()
    await ws.prepare(request)
    connected_clients.add(ws)
    print("Web client connected")
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    command = data.get("command")
                    print(f"Received command: {command}")
                    
                    if command == "start_live_calibration": 
                        await handle_start_live_calibration()
                    elif command == "stop_live_calibration": 
                        await handle_stop_live_calibration()
                    elif command == "build_reference": 
                        asyncio.create_task(handle_build_reference())
                    elif command == "capture_anchors": 
                        asyncio.create_task(handle_capture_anchors())
                    elif command == "calculate_placements": 
                        asyncio.create_task(handle_calculate_placements())
                    elif command == "capture_single": 
                        asyncio.create_task(handle_capture_single_final())
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                except Exception as e:
                    print(f"Error handling client message: {e}")
    except Exception as e:
        print(f"Client handler error: {e}")
    finally:
        connected_clients.discard(ws)
        print("Web client disconnected")
    return ws

async def handle_start_live_calibration():
    print("Starting live calibration...")
    try:
        shutil.rmtree(CALIBRATION_RUN_DIR, ignore_errors=True)
        os.makedirs(CALIBRATION_RUN_DIR, exist_ok=True)
        
        shared_state["live_stitchers"] = {
            esp_id: stitching_logic.MemoryEfficientStitcher() for esp_id in ESP_IPS
        }
        shared_state["is_live_calibrating"] = True
        
        for esp_id in ESP_IPS:
            await send_to_esp(esp_id, "start_capture")
        
        await broadcast_to_web_clients({"status": "live_calibration_started"})
        print("Live calibration started successfully")
    except Exception as e:
        print(f"Error starting live calibration: {e}")
        await broadcast_to_web_clients({"status": "live_calibration_error", "error": str(e)})

async def handle_stop_live_calibration():
    print("Stopping live calibration...")
    try:
        shared_state["is_live_calibrating"] = False
        shared_state["live_stitchers"].clear()
        
        for esp_id in ESP_IPS:
            await send_to_esp(esp_id, "stop_capture")
        
        await broadcast_to_web_clients({"status": "live_calibration_stopped"})
        print("Live calibration stopped successfully")
    except Exception as e:
        print(f"Error stopping live calibration: {e}")

async def handle_build_reference():
    try:
        await broadcast_to_web_clients({"status": "building_reference_started"})
        loop = asyncio.get_event_loop()
        ref_path = await loop.run_in_executor(None, stitching_logic.build_reference_panorama_efficient, CALIBRATION_RUN_DIR)
        if ref_path:
            await broadcast_to_web_clients({"status": "building_reference_complete", "path": ref_path})
            print(f"Reference panorama built: {ref_path}")
        else:
            await broadcast_to_web_clients({"status": "building_reference_failed"})
            print("Failed to build reference panorama")
    except Exception as e:
        print(f"Error building reference: {e}")
        await broadcast_to_web_clients({"status": "building_reference_failed", "error": str(e)})

async def handle_capture_anchors():
    try:
        await broadcast_to_web_clients({"status": "capturing_anchors_started"})
        
        shutil.rmtree(ANCHOR_FRAMES_DIR, ignore_errors=True)
        os.makedirs(ANCHOR_FRAMES_DIR, exist_ok=True)
        
        shared_state["anchor_frames_received"].clear()
        shared_state["anchor_capture_event"] = asyncio.Event()
        shared_state["is_capturing_anchors"] = True
        
        for esp_id in ESP_IPS:
            await send_to_esp(esp_id, "capture_single_frame")
        
        await asyncio.wait_for(shared_state["anchor_capture_event"].wait(), timeout=15.0)
        await broadcast_to_web_clients({"status": "capturing_anchors_complete"})
        print("Anchor capture completed successfully")
        
    except asyncio.TimeoutError:
        print("Anchor capture timed out")
        await broadcast_to_web_clients({"status": "capturing_anchors_failed", "error": "timeout"})
    except Exception as e:
        print(f"Error capturing anchors: {e}")
        await broadcast_to_web_clients({"status": "capturing_anchors_failed", "error": str(e)})
    finally:
        shared_state["is_capturing_anchors"] = False

async def handle_calculate_placements():
    try:
        await broadcast_to_web_clients({"status": "calculating_placements_started"})
        anchor_paths = {str(i+1): f"{ANCHOR_FRAMES_DIR}/anchor_{i+1}.jpg" for i in range(len(ESP_IPS))}
        loop = asyncio.get_event_loop()
        placements_path = await loop.run_in_executor(None, stitching_logic.calculate_placement_homographies, "data/reference_panorama.jpg", anchor_paths)
        if placements_path:
            await broadcast_to_web_clients({"status": "calculating_placements_complete", "path": placements_path})
            print(f"Placements calculated: {placements_path}")
        else:
            await broadcast_to_web_clients({"status": "calculating_placements_failed"})
            print("Failed to calculate placements")
    except Exception as e:
        print(f"Error calculating placements: {e}")
        await broadcast_to_web_clients({"status": "calculating_placements_failed", "error": str(e)})

async def handle_capture_single_final():
    try:
        capture_task = asyncio.create_task(handle_capture_anchors())
        await capture_task
        
        if len(shared_state["anchor_frames_received"]) != len(ESP_IPS):
            await broadcast_to_web_clients({
                "status": "capture_stitch_failed", 
                "reason": f"Did not receive all frames. Got {len(shared_state['anchor_frames_received'])}/{len(ESP_IPS)}"
            })
            return
        
        await broadcast_to_web_clients({"status": "stitching_final_frame"})
        frame_paths = [shared_state["anchor_frames_received"][str(i+1)] for i in range(len(ESP_IPS))]
        loop = asyncio.get_event_loop()
        stitched_path = await loop.run_in_executor(None, stitching_logic.stitch_from_placements, frame_paths, "data/camera_placements.npy", "data/reference_panorama.jpg")
        
        if stitched_path:
            await broadcast_to_web_clients({"status": "capture_stitch_complete", "stitched_image_path": stitched_path})
            print(f"Final stitch completed: {stitched_path}")
        else:
            await broadcast_to_web_clients({"status": "capture_stitch_failed"})
            print("Failed to stitch final image")
    except Exception as e:
        print(f"Error in capture single final: {e}")
        await broadcast_to_web_clients({"status": "capture_stitch_failed", "error": str(e)})

async def main():
    try:
        os.makedirs("data/final_stitched", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        await connect_to_esps()
        
        app = aiohttp.web.Application()
        app.add_routes([aiohttp.web.get("/bridge", client_handler)])
        runner = aiohttp.web.AppRunner(app)
        await runner.setup()
        site = aiohttp.web.TCPSite(runner, "127.0.0.1", 8080)
        await site.start()
        
        print("Server running at http://127.0.0.1:8080")
        print("Ready to receive frames from ESPs...")
        
        await asyncio.Event().wait()
        
    except Exception as e:
        print(f"Error in main: {e}")
        raise

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Fatal error: {e}")