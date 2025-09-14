#!/usr/bin/env python3
import asyncio
import aiohttp
import subprocess
import platform
import socket
import sys

async def debug_esp_connectivity():
    """Debug tool to test ESP32 connectivity issues"""
    
    # Your ESP IPs
    ESP_IPS = [
        "192.168.137.57",   # ESP1
        "192.168.137.193",  # ESP2  
        "192.168.137.129",  # ESP3
        "192.168.137.17",   # ESP3 (current working)
        "192.168.137.200",  # ESP4
    ]
    
    def get_local_ip_for_target(target_ip: str) -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((target_ip, 80))
                return s.getsockname()[0]
        except Exception:
            return "unknown"
    
    def sync_ping(ip: str, timeout: float = 2.0) -> tuple[bool, str]:
        system = platform.system().lower()
        if system == "windows":
            cmd = ["ping", "-n", "1", "-w", str(int(timeout * 1000)), ip]
        else:
            cmd = ["ping", "-c", "1", "-W", str(int(timeout)), ip]
        
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                timeout=timeout + 0.5, text=True)
            return proc.returncode == 0, f"stdout: {proc.stdout[:100]}..."
        except Exception as e:
            return False, f"exception: {e}"
    
    async def async_http_test(ip: str) -> tuple[bool, str]:
        try:
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://{ip}/", timeout=timeout) as resp:
                    return True, f"status: {resp.status}"
        except Exception as e:
            return False, f"error: {e}"
    
    async def async_ws_test(ip: str) -> tuple[bool, str]:
        try:
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(f"ws://{ip}/ws", timeout=timeout) as ws:
                    return True, "connected"
        except Exception as e:
            return False, f"error: {e}"
    
    print("=== ESP32 Connectivity Debug ===\n")
    
    for i, ip in enumerate(ESP_IPS, 1):
        print(f"ESP {i} ({ip}):")
        
        # Local routing info
        local_ip = get_local_ip_for_target(ip)
        print(f"  Local IP used: {local_ip}")
        
        # Ping test
        ping_ok, ping_msg = sync_ping(ip)
        print(f"  Ping: {'✓' if ping_ok else '✗'} - {ping_msg}")
        
        if ping_ok:  # Only test HTTP/WS if ping succeeds
            # HTTP test
            http_ok, http_msg = await async_http_test(ip)
            print(f"  HTTP: {'✓' if http_ok else '✗'} - {http_msg}")
            
            # WebSocket test
            ws_ok, ws_msg = await async_ws_test(ip)
            print(f"  WebSocket: {'✓' if ws_ok else '✗'} - {ws_msg}")
        else:
            print(f"  HTTP: skipped (ping failed)")
            print(f"  WebSocket: skipped (ping failed)")
        
        print()

if __name__ == "__main__":
    asyncio.run(debug_esp_connectivity())