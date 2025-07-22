import aiohttp
import aiohttp.web
import asyncio
import os
import json
from datetime import datetime
# --- NEW: Import your stitching logic ---
import stitching_logic

# --- Configuration ---
ESP_IPS = {
    '1': '192.168.137.239',  # esp32-257AC8
    '2': '192.168.137.74',  # esp32-24F7DC
    '3': '192.168.137.16', # esp32-255950 
}
ESP_WS_URL = "ws://{ip}/ws"
CALIBRATION_FRAME_COUNT = 15 # Set desired number of frames here

# --- Global State ---
connected_clients = set()
esp_websockets = {}
is_calibrating = False
# --- NEW: State for single frame capture ---
single_capture_frames = {}
# Create a dictionary to hold our shared state, including asyncio primitives
shared_state = {}


# --- WebSocket and UDP Handling (Modified) ---

async def connect_to_esps():
    """Establish WebSocket connections to all ESP32 modules."""
    for esp_id, ip in ESP_IPS.items():
        try:
            session = aiohttp.ClientSession()
            ws_url = ESP_WS_URL.format(ip=ip)
            # Add a timeout to the connection attempt
            ws = await asyncio.wait_for(session.ws_connect(ws_url), timeout=5.0)
            esp_websockets[esp_id] = (session, ws)
            print(f"Successfully connected to ESP {esp_id} at {ws_url}")
        except Exception as e:
            print(f"Failed to connect to ESP {esp_id}: {e}")

async def send_to_esp(esp_id, message):
    """Send a message to a specific ESP32."""
    if esp_id in esp_websockets:
        _, ws = esp_websockets[esp_id]
        if not ws.closed:
            try:
                await ws.send_str(message)
                print(f"Sent '{message}' to ESP {esp_id}")
            except Exception as e:
                print(f"Error sending to ESP {esp_id}: {e}")
        else:
            print(f"ESP {esp_id} WebSocket is closed.")
    else:
        print(f"ESP {esp_id} is not connected.")


async def broadcast_to_web_clients(message):
    """Broadcast a message to all connected web clients."""
    for ws in list(connected_clients):
        try:
            await ws.send_json(message)
        except Exception as e:
            print(f"Failed to send message to web client: {e}")

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
                    print(f"Received command from web client: {command}")

                    if command == "start_all_streams":
                        asyncio.create_task(handle_start_all_streams())
                    elif command == "stop_all_streams":
                        asyncio.create_task(handle_stop_all_streams())
                    elif command == "calibrate":
                        asyncio.create_task(handle_calibration())
                    elif command == "process_calibration":
                        asyncio.create_task(handle_process_calibration())
                    elif command == "calculate_homography":
                        asyncio.create_task(handle_calculate_homography())
                    # --- NEW: Handle single capture command ---
                    elif command == "capture_single":
                        asyncio.create_task(handle_capture_single())

                except json.JSONDecodeError:
                    print(f"Invalid JSON received: {msg.data}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        connected_clients.remove(ws)
        print("Web client disconnected")
    return ws

async def broadcast_image_to_web(image_data):
    for ws in list(connected_clients):
        try:
            await ws.send_bytes(image_data)
        except Exception as e:
            print(f"Failed to send image to client: {e}")

class UDPHandler(asyncio.DatagramProtocol):
    def __init__(self, port, shared_state):
        super().__init__()
        self.buffer = b""
        self.shared_state = shared_state # Store the shared state
        self.port = port
        self.timeout_task = None
        self.calibration_frame_count = 0
        self.is_capturing_single = False

    def connection_made(self, transport):
        self.transport = transport
        print(f"UDP server listening on port {self.port}")

    def datagram_received(self, data, addr):
        asyncio.create_task(self.process_datagram(data))

    async def process_datagram(self, data):
        self.buffer += data
        
        if b"IMAGE_END" in self.buffer:
            start_idx = self.buffer.find(b"IMAGE_START")
            end_idx = self.buffer.find(b"IMAGE_END")

            if start_idx != -1:
                image_data = self.buffer[start_idx + len(b"IMAGE_START"):end_idx]
                self.buffer = self.buffer[end_idx + len(b"IMAGE_END"):]

                esp_id = str(self.port)[-1]
                
                # --- MODIFIED: Handle different capture modes ---
                if self.is_capturing_single:
                    filename = f"data/single_frames/esp{esp_id}_capture.jpg"
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    with open(filename, 'wb') as f:
                        f.write(image_data)
                    single_capture_frames[esp_id] = filename
                    print(f"Saved single capture for ESP {esp_id}")
                    self.is_capturing_single = False
                    
                    if len(single_capture_frames) == len(ESP_IPS):
                        # Use the event from the shared state dictionary
                        self.shared_state['single_capture_event'].set()

                elif is_calibrating and self.calibration_frame_count < CALIBRATION_FRAME_COUNT:
                    self.calibration_frame_count += 1
                    filename = f"data/calibration/esp{esp_id}/frame_{self.calibration_frame_count:02d}.jpg"
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    with open(filename, 'wb') as f:
                        f.write(image_data)
                    
                    # --- CHANGE HERE: Use the constant ---
                    if self.calibration_frame_count == CALIBRATION_FRAME_COUNT:
                        await broadcast_to_web_clients({
                            "status": "calibration_capture_complete", 
                            "esp_id": esp_id
                        })
                        
                else:
                    # Default behavior: Just broadcast for live view
                    image_data_with_id = bytes(esp_id, 'utf-8') + image_data
                    await broadcast_image_to_web(image_data_with_id)

    def error_received(self, exc):
        print(f"Error received: {exc}")
        
# --- Global UDP Handlers dictionary ---
udp_handlers = {}

# --- Command Handling Logic (Updated) ---
async def handle_calibration():
    global is_calibrating
    is_calibrating = True
    for handler in udp_handlers.values():
        handler.calibration_frame_count = 0 
        
    print(f"Starting calibration, requesting {CALIBRATION_FRAME_COUNT} frames...")
    await broadcast_to_web_clients({"status": f"calibration_started_{CALIBRATION_FRAME_COUNT}_frames"})
    
    command_to_send = f"start_calibration:{CALIBRATION_FRAME_COUNT}"
    for esp_id in ESP_IPS:
        await send_to_esp(esp_id, command_to_send)

async def handle_process_calibration():
    print("Processing calibration images...")
    await broadcast_to_web_clients({"status": "processing_started"})
    
    loop = asyncio.get_event_loop()
    panorama_paths = {}
    
    for esp_id in ESP_IPS:
        # --- USE YOUR STITCHING LOGIC ---
        path = await loop.run_in_executor(None, stitching_logic.stitch_calibration_images, esp_id)
        if path:
            panorama_paths[esp_id] = path

    print("All panoramas stitched.")
    await broadcast_to_web_clients({
        "status": "processing_complete",
        "panorama_paths": panorama_paths
    })
    global is_calibrating
    is_calibrating = False

async def handle_calculate_homography():
    print("Calculating homography...")
    await broadcast_to_web_clients({"status": "homography_started"})
    
    panorama_paths = [f"data/stitched_panoramas/esp{i}_panorama.jpg" for i in ESP_IPS.keys()]
    
    loop = asyncio.get_event_loop()
    # --- USE YOUR STITCHING LOGIC ---
    final_image, matrices_path = await loop.run_in_executor(None, stitching_logic.calculate_homography_and_stitch, panorama_paths)
    
    if final_image and matrices_path:
        print("Homography calculation complete.")
        await broadcast_to_web_clients({
            "status": "homography_complete",
            "final_image_path": final_image
        })
    else:
        print("Homography calculation failed.")
        await broadcast_to_web_clients({"status": "homography_failed"})

# --- NEW: Single Capture Handler ---
async def handle_capture_single():
    # This function now correctly accesses the global shared_state
    global single_capture_frames 
    print("Starting single frame capture...")
    
    single_capture_frames.clear()
    
    # Access the event from the shared state and clear it for this new operation
    capture_event = shared_state['single_capture_event']
    capture_event.clear()
    
    # Tell UDP handlers to save the next frame
    for handler in udp_handlers.values():
        handler.is_capturing_single = True
        
    # Tell ESPs to send one frame
    for esp_id in ESP_IPS:
        await send_to_esp(esp_id, "capture_single_frame")
        
    # Wait for all 3 frames to be saved
    try:
        # Wait on the event from the shared state
        await asyncio.wait_for(capture_event.wait(), timeout=10.0)
    except asyncio.TimeoutError:
        print("Timeout waiting for single frames.")
        await broadcast_to_web_clients({"status": "capture_failed_timeout"})
        return
        
    print("All single frames received. Stitching...")
    await broadcast_to_web_clients({"status": "stitching_single_frames"})
    
    frame_paths = [single_capture_frames[id] for id in sorted(single_capture_frames.keys())]
    
    loop = asyncio.get_event_loop()
    # --- USE YOUR FAST STITCHING LOGIC ---
    stitched_path = await loop.run_in_executor(None, stitching_logic.stitch_single_frames, frame_paths)
    
    if stitched_path:
        await broadcast_to_web_clients({
            "status": "capture_stitch_complete",
            "stitched_image_path": stitched_path
        })
    else:
        await broadcast_to_web_clients({"status": "capture_stitch_failed"})

# --- Main Application Setup ---
async def main():
    os.makedirs("data/calibration/esp1", exist_ok=True)
    os.makedirs("data/calibration/esp2", exist_ok=True)
    os.makedirs("data/calibration/esp3", exist_ok=True)
    os.makedirs("data/stitched_panoramas", exist_ok=True)
    os.makedirs("data/final_stitched", exist_ok=True)
    os.makedirs("data/single_frames", exist_ok=True)

    await connect_to_esps()

    loop = asyncio.get_event_loop()

    # Now that the loop is running, we can safely create the event
    shared_state['single_capture_event'] = asyncio.Event()
    
    for i, port in enumerate([10101, 10102, 10103], 1):
        # Store handler instances to modify their state
        handler = UDPHandler(port, shared_state)
        udp_handlers[str(i)] = handler
        await loop.create_datagram_endpoint(lambda: handler, local_addr=("0.0.0.0", port))
        print(f"UDP server started on 0.0.0.0:{port}")

    app = aiohttp.web.Application()
    app.add_routes([aiohttp.web.get("/bridge", client_handler)])
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, host="127.0.0.1", port=8080)
    await site.start()
    print("WebSocket server started on http://127.0.0.1:8080/bridge")

    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
        for session, ws in esp_websockets.values():
            await session.close()

# --- NEW: Command Handlers for Live Streaming Test ---
async def handle_start_all_streams():
    """Sends the 'start_capture' command to all connected ESPs."""
    print("Broadcasting start stream command to all ESPs...")
    await broadcast_to_web_clients({"status": "Live streaming started."})
    for esp_id in ESP_IPS:
        # The ESPs already understand 'start_capture' for continuous streaming
        await send_to_esp(esp_id, "start_capture")

async def handle_stop_all_streams():
    """Sends the 'stop_capture' command to all connected ESPs."""
    print("Broadcasting stop stream command to all ESPs...")
    await broadcast_to_web_clients({"status": "Live streaming stopped."})
    for esp_id in ESP_IPS:
        await send_to_esp(esp_id, "stop_capture")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down.")