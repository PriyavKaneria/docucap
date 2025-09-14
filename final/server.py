import aiohttp
import aiohttp.web
import asyncio
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import shutil
import cv2
import numpy as np

# Make sure we can import the pipeline from this folder
BASE_DIR = Path(__file__).parent.resolve()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from panorama_pipeline import PanoramaPipeline, PipelineConfig
# Allow importing esp_status_monitor from project root
PARENT_DIR = BASE_DIR.parent.resolve()
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
try:
    from esp_status_monitor import ESP_MAP as MONITOR_ESP_MAP, run_arp as monitor_run_arp, parse_arp as monitor_parse_arp, ping_once as monitor_ping_once, REFRESH_INTERVAL_S as MONITOR_REFRESH_INTERVAL
except Exception:
    MONITOR_ESP_MAP = {}
    def monitor_run_arp(): return ""
    def monitor_parse_arp(_): return []
    def monitor_ping_once(_ip, timeout=1.0): return False
    MONITOR_REFRESH_INTERVAL = 2.0

# ---------------- Configuration ----------------
def env_or_default(key: str, default: str) -> str:
    return os.environ.get(key, default)

# 4 ESPs by default. Override via environment variables ESP1_IP..ESP4_IP if needed.
ESP_IDS = [s.strip() for s in os.environ.get("ESP_IDS", "1,2,3,4").split(",") if s.strip()]
ESP_WS_URL = "ws://{ip}/ws"
ESP_WS_PORTS = [80, 81]
ESP_WS_PATHS = ["/ws", "/socket", "/ws/"]

# Optional MAC-based mapping (recommended). You can set these via environment vars
# ESP1_MAC..ESP4_MAC or derive from esp_status_monitor. If not set, we fall back to
# static ESP_IPS and dynamic ARP mapping heuristics.
ESP_MACS = {
    "1": "c82e18255950",
    "2": "c82e1824f7dc",
    "3": "c82e1823124c",
    "4": "c82e18257ac8",
}
# Internal resolved map id -> mac
ESP_ID_TO_MAC: dict[str, str] = {}

def _build_id_to_mac_from_monitor_map():
    # Derive id->mac from MONITOR_ESP_MAP names if they include esp1/2/3/4 (case-insensitive)
    for mac, name in MONITOR_ESP_MAP.items():
        n = (name or "").lower()
        for i in range(1, 5):
            if f"esp{i}" in n or n.endswith(str(i)) or f" {i}" in n:
                ESP_ID_TO_MAC.setdefault(str(i), mac.lower())

    # Environment overrides take priority
    for i in range(1, 5):
        env_mac = ESP_MACS.get(str(i), "")
        if env_mac:
            ESP_ID_TO_MAC[str(i)] = env_mac.lower()

_build_id_to_mac_from_monitor_map()

def _derive_ids_from_mapping():
    global ESP_IDS
    if ESP_ID_TO_MAC:
        # Do not shrink the managed set when mapping is partial; merge with existing ids.
        existing = set(ESP_IDS)
        mapped = set(ESP_ID_TO_MAC.keys())
        merged = existing | mapped
        try:
            ESP_IDS = sorted(merged, key=lambda x: int(x))
        except Exception:
            ESP_IDS = list(sorted(merged))

_derive_ids_from_mapping()

def debug_dump_config():
    try:
        print(f"[Config] ESP_IDS={ESP_IDS}")
        print(f"[Config] ESP_ID_TO_MAC={ESP_ID_TO_MAC}")
        for eid in ESP_IDS:
            print(f"[Config] ENV ESP{eid}_IP={os.environ.get(f'ESP{eid}_IP','')}")
    except Exception as e:
        print(f"[Config] dump error: {e}")

def env_ip_for_id(esp_id: str) -> str | None:
    return os.environ.get(f"ESP{esp_id}_IP")

def resolve_ip_for_esp(esp_id: str) -> str | None:
    """
    Resolve current IP for a given ESP id using:
    1) MAC mapping from ESP_ID_TO_MAC against monitor_state
    2) Fallback to ESP{N}_IP env var if no MAC/IP known
    """
    mac = ESP_ID_TO_MAC.get(esp_id, "")
    if mac:
        rec = monitor_state["devices"].get(mac)
        if rec and rec.get("ip"):
            return rec["ip"]
    # Fallback to environment override
    return env_ip_for_id(esp_id)

# Data dirs (inside final/)
DATA_DIR = BASE_DIR / "data"
CALIBRATION_RUN_DIR = DATA_DIR / "calibration_run"
ANCHOR_FRAMES_DIR = DATA_DIR / "anchors"
PREVIEW_FRAMES_DIR = DATA_DIR / "previews"

# ---------------- Global State ----------------
connected_clients = set()
esp_websockets: dict[str, tuple[aiohttp.ClientSession, aiohttp.ClientWebSocketResponse]] = {}
# Per-ESP connection and listener task registries for GC and auto-reconnect
esp_connect_tasks: dict[str, asyncio.Task] = {}
esp_listen_tasks: dict[str, asyncio.Task] = {}
# Reusable aiohttp sessions per ESP to avoid leaking/creating many sessions
esp_sessions: dict[str, aiohttp.ClientSession] = {}
# Per-ESP periodic capture loops for live calibration (WS command "capture")
esp_capture_loops: dict[str, asyncio.Task] = {}
state = {
    "is_live_calibrating": False,
    "is_capturing_anchors": False,
    "anchor_capture_event": None,
    "anchor_frames_received": {},    # keys "esp_1".. "esp_4" -> path
    "is_capturing_preview": False,
    "preview_capture_event": None,
    "preview_frames_received": {},
    "last_reference_panorama": None, # str path
    "last_calibration_file": None,   # str path
    "settings": {                    # defaults, may be updated per request
        "detector": "orb",
        "features": 5000,
        "exposure": True,
        "seams": True,
    },
}

# Monitor state for navbar status
monitor_state = {"devices": {}}

# Single pipeline instance reused (settings may be re-applied on demand)
def create_pipeline_from_state() -> PanoramaPipeline:
    cfg = PipelineConfig()
    cfg.DETECTOR_TYPE = state["settings"]["detector"]
    cfg.N_FEATURES = int(state["settings"]["features"])
    cfg.PERFORM_EXPOSURE_COMPENSATION = bool(state["settings"]["exposure"])
    cfg.PERFORM_SEAM_FINDING = bool(state["settings"]["seams"])
    # 4 cameras by default
    cfg.ESP_CAMERAS = ["esp_1", "esp_2", "esp_3", "esp_4"]
    return PanoramaPipeline(cfg)

# ---------------- WebSocket to ESPs ----------------
def start_esp_connection_manager():
    """
    Spawn a persistent connection task per ESP that auto-reconnects and
    reattaches listeners whenever a device restarts or is unplugged.
    """
    # Refresh id->mac mapping at startup
    _build_id_to_mac_from_monitor_map()
    print(f"[Server] Managing ESP ids: {ESP_IDS}")
    for esp_id in ESP_IDS:
        if esp_id in esp_connect_tasks and not esp_connect_tasks[esp_id].done():
            continue
        esp_connect_tasks[esp_id] = asyncio.create_task(ensure_esp_connection(esp_id))
        print(f"[Server] Spawned connection task for ESP {esp_id}")

async def ensure_esp_connection(esp_id: str):
    """
    Forever loop: ensure we have a live websocket to the ESP (resolving its IP dynamically),
    attach a listener, and on any error/disconnect, clean up and retry with backoff.
    Reuses a single ClientSession per ESP.
    """
    backoff = 0.5
    while True:
        # Do NOT tear down healthy connections each cycle. Only evaluate and clean when needed.
        # Cache current handles; conditional cleanup happens after reachability checks.
        t = esp_listen_tasks.get(esp_id)
        tup = esp_websockets.get(esp_id)

        ws = None
        # print(f"[ESP {esp_id}] ensure loop tick; backoff={backoff}")
        try:
            # Resolve current target IP dynamically (from monitor or static fallback)
            ip = resolve_ip_for_esp(esp_id)
            if not ip:
                # print(f"[ESP {esp_id}] no IP resolved yet; waiting for monitor to resolve; will retry")
                raise ConnectionError("no_ip_resolved")
            # print(f"[ESP {esp_id}] attempting socket connection to {ip}...")

            # Ping gate: prefer ping but tolerate ICMP being blocked by also probing HTTP root.
            loop = asyncio.get_event_loop()
            try:
                ping_ok = await loop.run_in_executor(None, monitor_ping_once, ip)
            except Exception:
                ping_ok = False

            # Get or create a reusable session for this ESP
            session = esp_sessions.get(esp_id)
            if session is None or session.closed:
                session = aiohttp.ClientSession()
                esp_sessions[esp_id] = session

            http_ok = False
            try:
                # Small timeout to quickly skip dead hosts; some sketches don't serve "/" which is fine.
                async with session.get(f"http://{ip}/", timeout=aiohttp.ClientTimeout(total=2.0)) as resp:
                    http_ok = (resp.status < 600)
            except Exception:
                http_ok = False

            reachable = ping_ok or http_ok

            # Determine current listener/ws state
            listen_alive = bool(t) and not t.done()
            existing_open = False
            ws0 = None
            if tup:
                _sess0, ws0 = tup
                existing_open = ws0 is not None and not ws0.closed
            
            # print(f"[ESP {esp_id}] reachability: ping={ping_ok}, http={http_ok}, listener_alive={listen_alive}, ws_open={existing_open}")

            if not reachable:
                # If we had an active connection, clean it up; otherwise just back off and retry later
                if listen_alive:
                    try:
                        t.cancel()
                        await t
                    except:
                        pass
                    esp_listen_tasks.pop(esp_id, None)
                if existing_open:
                    try:
                        await ws0.close()
                    except:
                        pass
                    esp_websockets.pop(esp_id, None)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 10.0)
                continue

            print(f"[ESP {esp_id}] reachable at {ip} (ping={ping_ok}, http={http_ok})")
            # If reachable and we already have an active listener and open ws, keep it
            if listen_alive and existing_open:
                backoff = 0.5
                await asyncio.sleep(1.0)
                continue

            # Try multiple common websocket endpoints (ports/paths)
            connected = False
            last_err: Exception | None = None
            for port in ESP_WS_PORTS:
                for path in ESP_WS_PATHS:
                    # Build ws URL
                    if port == 80:
                        url = f"ws://{ip}{path}"
                    else:
                        url = f"ws://{ip}:{port}{path}"
                    try:
                        ws = await session.ws_connect(url, timeout=5.0, heartbeat=20.0)
                        esp_websockets[esp_id] = (session, ws)
                        print(f"[ESP {esp_id}] connected to {url}")
                        await broadcast_to_web_clients({"status": "esp_ws_connected", "esp_id": esp_id})
                        connected = True
                        break
                    except Exception as _e:
                        last_err = _e
                        continue
                if connected:
                    break

            if not connected:
                # Exhausted all endpoints
                raise ConnectionError(f"ws_connect_failed_all_endpoints: {type(last_err).__name__}: {str(last_err) if last_err else ''}")

            # Start listener and wait for it to finish
            listen_task = asyncio.create_task(listen_to_esp(esp_id, ws))
            esp_listen_tasks[esp_id] = listen_task
            try:
                await listen_task
            finally:
                # Remove task reference when done
                esp_listen_tasks.pop(esp_id, None)

            print(f"[ESP {esp_id}] listener ended")
            await broadcast_to_web_clients({"status": "esp_ws_disconnected", "esp_id": esp_id})
            # Successful cycle; reset backoff for faster future reconnects
            backoff = 0.5

        except asyncio.CancelledError:
            # Manager is being shut down; close ws and session, clean registries, then exit loop
            try:
                if ws and not ws.closed:
                    await ws.close()
            except:
                pass
            sess = esp_sessions.pop(esp_id, None)
            if sess and not sess.closed:
                try:
                    await sess.close()
                except:
                    pass
            esp_websockets.pop(esp_id, None)
            t = esp_listen_tasks.pop(esp_id, None)
            if t and not t.done():
                t.cancel()
                try:
                    await t
                except:
                    pass
            break
        except Exception as e:
            # Connection or handshake error; ensure ws is closed, keep session for reuse
            try:
                if ws and not ws.closed:
                    await ws.close()
            except:
                pass
            # print(f"[ESP {esp_id}] connection error: {e.__class__.__name__}: {str(e) or repr(e)}")
        finally:
            # Ensure mapping doesn't hold stale ws
            tup = esp_websockets.get(esp_id)
            if tup:
                _sess, cur_ws = tup
                if cur_ws is ws:
                    esp_websockets.pop(esp_id, None)

        await asyncio.sleep(backoff)
        backoff = min(backoff * 2.0, 10.0)

async def listen_to_esp(esp_id: str, ws: aiohttp.ClientWebSocketResponse):
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                await handle_esp_binary_message(esp_id, msg.data)
    except Exception as e:
        print(f"Error listening to ESP {esp_id}: {e}")
    finally:
        print(f"ESP {esp_id} disconnected.")

async def handle_esp_binary_message(esp_id: str, image_data: bytes):
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"ESP {esp_id}: Failed to decode JPEG (len={len(image_data)})")
            return
        print("ESP {}: Received frame {}x{}".format(esp_id, frame.shape[1], frame.shape[0]))

        if state["is_live_calibrating"]:
            # Save all frames to rotation sweep dir
            ts = datetime.now().timestamp()
            out_path = CALIBRATION_RUN_DIR / f"{int(ts*1000)}_esp_{esp_id}.jpg"
            CALIBRATION_RUN_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), frame)
        elif state["is_capturing_anchors"]:
            # Save one frame for each ESP (first frame we receive after trigger)
            key = f"esp_{esp_id}"
            if key not in state["anchor_frames_received"]:
                ANCHOR_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
                out_path = ANCHOR_FRAMES_DIR / f"anchor_{key}.jpg"
                cv2.imwrite(str(out_path), frame)
                state["anchor_frames_received"][key] = str(out_path)
                print(f"Saved anchor for {key} -> {out_path}")
                if len(state["anchor_frames_received"]) == len(ESP_IDS) and state["anchor_capture_event"]:
                    state["anchor_capture_event"].set()
        elif state["is_capturing_preview"]:
            # Save one frame per ESP for preview
            key = f"esp_{esp_id}"
            if key not in state["preview_frames_received"]:
                PREVIEW_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
                out_path = PREVIEW_FRAMES_DIR / f"preview_{key}.jpg"
                cv2.imwrite(str(out_path), frame)
                state["preview_frames_received"][key] = str(out_path)
                if len(state["preview_frames_received"]) == len(ESP_IDS) and state["preview_capture_event"]:
                    state["preview_capture_event"].set()

        # Always forward the live frame to UI with a leading ID byte
        message_data = bytes(esp_id, "utf-8") + image_data
        await broadcast_bytes_to_web_clients(message_data)
    except Exception as e:
        print(f"ESP {esp_id}: Error processing frame: {e}")

async def send_to_esp(esp_id: str, message: str):
    if esp_id in esp_websockets and not esp_websockets[esp_id][1].closed:
        await esp_websockets[esp_id][1].send_str(message)

# WS-command capture loops (compatible with mvpV0.ino)
async def capture_loop(esp_id: str, interval_s: float = 0.35):
    """
    Periodically sends 'capture' over WebSocket to the ESP while live calibration is active.
    mvpV0.ino responds to 'capture' by grabbing one frame and pushing it over WS.
    """
    try:
        while state["is_live_calibrating"]:
            await send_to_esp(esp_id, "capture")
            await asyncio.sleep(interval_s)
    except asyncio.CancelledError:
        pass

async def start_capture_loops():
    for esp_id in ESP_IDS:
        if esp_id in esp_capture_loops and not esp_capture_loops[esp_id].done():
            continue
        esp_capture_loops[esp_id] = asyncio.create_task(capture_loop(esp_id))

async def stop_capture_loops():
    for esp_id, task in list(esp_capture_loops.items()):
        if task and not task.done():
            task.cancel()
            try:
                await task
            except:
                pass
        esp_capture_loops.pop(esp_id, None)

# ---------------- Web client bridge ----------------
async def broadcast_to_web_clients(message: dict):
    for ws in list(connected_clients):
        try:
            await ws.send_json(message)
        except Exception as e:
            print(f"Error broadcasting JSON to client: {e}")
            connected_clients.discard(ws)

async def broadcast_bytes_to_web_clients(data: bytes):
    for ws in list(connected_clients):
        try:
            await ws.send_bytes(data)
        except Exception as e:
            print(f"Error broadcasting BYTES to client: {e}")
            connected_clients.discard(ws)

async def client_handler(request):
    ws = aiohttp.web.WebSocketResponse()
    await ws.prepare(request)
    connected_clients.add(ws)
    print("Web client connected")

    # Notify UI about server readiness
    await ws.send_json({"status": "ready", "esp_count": len(ESP_IDS)})

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except Exception:
                    data = {"command": msg.data}

                command = data.get("command")
                settings = data.get("settings") or {}
                print(f"Received command: {command}, settings: {settings}")

                # Merge settings when provided
                if settings:
                    if "detector" in settings:
                        state["settings"]["detector"] = settings["detector"]
                    if "features" in settings:
                        state["settings"]["features"] = int(settings["features"])
                    if "exposure" in settings:
                        state["settings"]["exposure"] = bool(settings["exposure"])
                    if "seams" in settings:
                        state["settings"]["seams"] = bool(settings["seams"])

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
                    asyncio.create_task(handle_capture_and_stitch_final())
                elif command == "capture_preview":
                    asyncio.create_task(handle_capture_preview())
                elif command == "ping":
                    await ws.send_json({"status": "pong"})
            elif msg.type == aiohttp.WSMsgType.ERROR:
                print(f"Web client connection closed with exception {ws.exception()}")
    finally:
        connected_clients.discard(ws)
        print("Web client disconnected")
    return ws

# ---------------- Workflow Handlers ----------------
async def handle_start_live_calibration():
    try:
        shutil.rmtree(CALIBRATION_RUN_DIR, ignore_errors=True)
        CALIBRATION_RUN_DIR.mkdir(parents=True, exist_ok=True)

        state["is_live_calibrating"] = True
        # Start periodic WS 'capture' loops per ESP for continuous frames
        await start_capture_loops()

        await broadcast_to_web_clients({"status": "live_calibration_started"})
    except Exception as e:
        await broadcast_to_web_clients({"status": "live_calibration_error", "error": str(e)})

async def handle_stop_live_calibration():
    try:
        state["is_live_calibrating"] = False
        # Stop periodic WS 'capture' loops
        await stop_capture_loops()
        await broadcast_to_web_clients({"status": "live_calibration_stopped"})
    except Exception as e:
        print(f"Error stopping live calibration: {e}")

async def handle_build_reference():
    try:
        await broadcast_to_web_clients({"status": "building_reference_started"})

        def _build():
            pipeline = create_pipeline_from_state()
            return pipeline.step1_generate_panorama(str(CALIBRATION_RUN_DIR))

        loop = asyncio.get_event_loop()
        ref_path = await loop.run_in_executor(None, _build)

        if ref_path:
            state["last_reference_panorama"] = ref_path
            rel = os.path.relpath(ref_path, BASE_DIR).replace("\\", "/")
            await broadcast_to_web_clients({"status": "building_reference_complete", "path": rel})
        else:
            await broadcast_to_web_clients({"status": "building_reference_failed"})
    except Exception as e:
        await broadcast_to_web_clients({"status": "building_reference_failed", "error": str(e)})

async def handle_capture_preview():
    try:
        await broadcast_to_web_clients({"status": "capture_preview_started"})

        # Reset preview directory
        shutil.rmtree(PREVIEW_FRAMES_DIR, ignore_errors=True)
        PREVIEW_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

        # Reset preview state
        state["preview_frames_received"].clear()
        state["preview_capture_event"] = asyncio.Event()
        state["is_capturing_preview"] = True

        # Trigger one frame from each ESP; frames will be forwarded and saved
        for esp_id in ESP_IDS:
            await send_to_esp(esp_id, "capture")

        # Wait briefly for all previews (non-fatal if some miss)
        try:
            await asyncio.wait_for(state["preview_capture_event"].wait(), timeout=10.0)
        except asyncio.TimeoutError:
            pass

        # Build relative paths for UI
        frames = {}
        for idx in range(1, len(ESP_IDS) + 1):
            key = f"esp_{idx}"
            p = state["preview_frames_received"].get(key)
            if p:
                frames[str(idx)] = os.path.relpath(p, BASE_DIR).replace("\\", "/")

        await broadcast_to_web_clients({"status": "capture_preview_complete", "frames": frames})
    except Exception as e:
        await broadcast_to_web_clients({"status": "capture_preview_failed", "error": str(e)})
    finally:
        state["is_capturing_preview"] = False

async def handle_capture_anchors():
    try:
        await broadcast_to_web_clients({"status": "capturing_anchors_started"})

        # Clear anchors
        shutil.rmtree(ANCHOR_FRAMES_DIR, ignore_errors=True)
        ANCHOR_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

        # Reset state
        state["anchor_frames_received"].clear()
        state["anchor_capture_event"] = asyncio.Event()
        state["is_capturing_anchors"] = True

        # Ask each ESP for a single frame
        for esp_id in ESP_IDS:
            await send_to_esp(esp_id, "capture")

        # Wait up to 15s for all
        await asyncio.wait_for(state["anchor_capture_event"].wait(), timeout=15.0)

        await broadcast_to_web_clients({"status": "capturing_anchors_complete"})
    except asyncio.TimeoutError:
        await broadcast_to_web_clients({"status": "capturing_anchors_failed", "error": "timeout"})
    except Exception as e:
        await broadcast_to_web_clients({"status": "capturing_anchors_failed", "error": str(e)})
    finally:
        state["is_capturing_anchors"] = False

async def handle_calculate_placements():
    try:
        await broadcast_to_web_clients({"status": "calculating_placements_started"})

        if not state["last_reference_panorama"]:
            await broadcast_to_web_clients({"status": "calculating_placements_failed", "error": "no_reference"})
            return

        # Build anchors map ordered by esp_1..esp_4
        anchors = {}
        for idx in range(1, len(ESP_IDS) + 1):
            key = f"esp_{idx}"
            path = state["anchor_frames_received"].get(key)
            if not path:
                await broadcast_to_web_clients({"status": "calculating_placements_failed", "error": f"missing_{key}"})
                return
            anchors[key] = path

        def _calc():
            pipeline = create_pipeline_from_state()
            calib_path = pipeline.step2_extract_from_anchors(anchors, state["last_reference_panorama"])
            return calib_path

        loop = asyncio.get_event_loop()
        calib_path = await loop.run_in_executor(None, _calc)
        state["last_calibration_file"] = calib_path

        rel_calib = os.path.relpath(calib_path, BASE_DIR).replace("\\", "/")
        await broadcast_to_web_clients({"status": "calculating_placements_complete", "calibration_file": rel_calib})
    except Exception as e:
        await broadcast_to_web_clients({"status": "calculating_placements_failed", "error": str(e)})

async def handle_capture_and_stitch_final():
    try:
        # First capture new anchors
        await handle_capture_anchors()
        if len(state["anchor_frames_received"]) != len(ESP_IDS):
            await broadcast_to_web_clients({
                "status": "capture_stitch_failed",
                "error": f"anchors_incomplete_{len(state['anchor_frames_received'])}/{len(ESP_IDS)}",
            })
            return

        if not state["last_calibration_file"]:
            await broadcast_to_web_clients({"status": "capture_stitch_failed", "error": "no_calibration"})
            return

        await broadcast_to_web_clients({"status": "stitching_final_frame"})

        # Order images as esp_1..esp_4
        ordered_images = []
        for idx in range(1, len(ESP_IDS) + 1):
            key = f"esp_{idx}"
            ordered_images.append(state["anchor_frames_received"][key])

        def _stitch():
            pipeline = create_pipeline_from_state()
            out_path = pipeline.step3_stitch_images(ordered_images, state["last_calibration_file"])
            return out_path

        loop = asyncio.get_event_loop()
        stitched_path = await loop.run_in_executor(None, _stitch)

        rel = os.path.relpath(stitched_path, BASE_DIR).replace("\\", "/")
        await broadcast_to_web_clients({"status": "capture_stitch_complete", "stitched_image_path": rel})
    except Exception as e:
        await broadcast_to_web_clients({"status": "capture_stitch_failed", "error": str(e)})

# ---------------- ESP Status Monitor Loop ----------------
async def monitor_esps_loop():
    """
    Periodically run ARP + ping (using esp_status_monitor) and broadcast status to UI.
    """
    loop = asyncio.get_event_loop()
    # seed known devices from mapping
    for mac, name in MONITOR_ESP_MAP.items():
        if mac not in monitor_state["devices"]:
            monitor_state["devices"][mac] = {
                "mac": mac, "name": name, "ip": "", "online": False, "last_seen": None
            }

    while True:
        try:
            out = await loop.run_in_executor(None, monitor_run_arp)
            found = monitor_parse_arp(out) if out is not None else []

            # Update entries and ping concurrently
            ping_tasks = []
            ping_macs = []
            now_str = datetime.now().strftime("%H:%M:%S")

            # Ensure mapping devices are present
            for mac, name in MONITOR_ESP_MAP.items():
                monitor_state["devices"].setdefault(mac, {"mac": mac, "name": name, "ip": "", "online": False, "last_seen": None})

            for (name, ip, mac) in found:
                rec = monitor_state["devices"].setdefault(mac, {"mac": mac, "name": name, "ip": "", "online": False, "last_seen": None})
                rec["name"] = name
                rec["ip"] = ip
                rec["last_seen"] = now_str
                ping_tasks.append(loop.run_in_executor(None, monitor_ping_once, ip))
                ping_macs.append(mac)

            if ping_tasks:
                results = await asyncio.gather(*ping_tasks, return_exceptions=True)
                for idx, mac in enumerate(ping_macs):
                    ok = False if isinstance(results[idx], Exception) else bool(results[idx])
                    monitor_state["devices"][mac]["online"] = ok

            # Prepare payload ordered by name
            devices_payload = []
            for mac, rec in monitor_state["devices"].items():
                devices_payload.append({
                    "mac": mac,
                    "name": rec.get("name", MONITOR_ESP_MAP.get(mac, mac)),
                    "ip": rec.get("ip", ""),
                    "online": bool(rec.get("online", False)),
                    "last_seen": rec.get("last_seen") or "-"
                })

            # Filter to mapped devices if mapping provided
            if MONITOR_ESP_MAP:
                devices_payload = [d for d in devices_payload if d["mac"] in MONITOR_ESP_MAP]
                devices_payload.sort(key=lambda d: d["name"])

            await broadcast_to_web_clients({"status": "esp_status", "devices": devices_payload})
        except Exception:
            # ignore errors; retry next cycle
            pass

        try:
            interval = float(MONITOR_REFRESH_INTERVAL)
        except Exception:
            interval = 2.0
        await asyncio.sleep(max(0.5, interval))

# ---------------- App bootstrap ----------------
async def watch_tasks_loop():
    try:
        while True:
            active = {}
            for eid in ESP_IDS:
                active[eid] = {
                    "connect": bool(esp_connect_tasks.get(eid)) and not esp_connect_tasks[eid].done(),
                    "listen": bool(esp_listen_tasks.get(eid)) and not esp_listen_tasks[eid].done(),
                    "ws": bool(esp_websockets.get(eid)) and not esp_websockets[eid][1].closed if esp_websockets.get(eid) else False,
                }
            print(f"[Watch] tasks={active}")
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        pass

async def make_app():
    # Ensure data dir exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CALIBRATION_RUN_DIR.mkdir(parents=True, exist_ok=True)
    ANCHOR_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Start status monitor loop first so ARP/IP info populates ASAP
    asyncio.create_task(monitor_esps_loop())

    # Optional debug dumps
    if os.environ.get("DEBUG_SERVER"):
        debug_dump_config()
    if os.environ.get("DEBUG_TASKS"):
        asyncio.create_task(watch_tasks_loop())

    # Then start the ESP connection manager (uses dynamic IP resolution)
    start_esp_connection_manager()

    app = aiohttp.web.Application()
    app.add_routes([aiohttp.web.get("/bridge", client_handler)])
    return app

async def main():
    app = await make_app()
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "127.0.0.1", 8080)
    await site.start()

    print(f"Server running at http://127.0.0.1:8080")
    print("Ready to receive frames from ESPs...")

    # Keep running
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Fatal error: {e}")
