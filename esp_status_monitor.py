#!/usr/bin/env python3
"""
ESP32 ARP Monitor with Ping Status
- Continuously parses `arp -a` to discover devices by MAC address.
- User provides a mapping of MAC -> Friendly Name (e.g., ESP32-1, ESP32-2).
- Displays live table with Name, IP, MAC, Status (ping), Last Seen.


Dependencies:
- Python 3.8+
- rich (pip install rich)
"""


import re
import time
import subprocess
from datetime import datetime
from typing import Dict, Optional
import platform


from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.console import Console
from rich import box
from rich.text import Text


console = Console()

# User-defined mapping: MAC (lowercase, no separators) -> Friendly Name
ESP_MAP: Dict[str, str] = {
    "c82e18255950": "esp32-255950",
    "c82e1824f7dc": "esp32-24f7dc",
    "c82e1823124c": "esp32-23124c",
    "c82e18257ac8": "esp32-257ac8",
}


REFRESH_INTERVAL_S = 2.0
PING_TIMEOUT_S = 1.0

class Device:
    def __init__(self, mac: str, ip: str, name: str):
        self.mac = mac
        self.ip = ip
        self.name = name
        self.last_seen: Optional[datetime] = None
        self.online: bool = False

    def update_seen(self):
        self.last_seen = datetime.now()

    def update_status(self):
        self.online = ping_once(self.ip, timeout=PING_TIMEOUT_S)


def run_arp() -> str:
    try:
        proc = subprocess.run(["arp", "-a"], capture_output=True, text=True, timeout=3)
        return proc.stdout
    except Exception:
        return ""


def parse_arp(out: str):
    devices = []
    for line in out.splitlines():
        ip = None
        mac = None

        m = re.search(r"\((\d+\.\d+\.\d+\.\d+)\).*?((?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2})", line, re.I)
        if m:
            ip, mac = m.group(1), m.group(2)
        else:
            m = re.search(r"(\d+\.\d+\.\d+\.\d+)\s+([0-9a-f\-]{17})", line, re.I)
            if m:
                ip, mac = m.group(1), m.group(2)

        if ip and mac:
            mac_norm = re.sub(r"[^0-9a-f]", "", mac.lower())
            if mac_norm in ESP_MAP:
                devices.append((ESP_MAP[mac_norm], ip, mac_norm))
    return devices


def ping_once(ip: str, timeout: float = 1.0) -> bool:
    system = platform.system().lower()
    if system == "windows":
        cmd = ["ping", "-n", "1", "-w", str(int(timeout * 1000)), ip]
    else:
        cmd = ["ping", "-c", "1", "-W", str(int(timeout)), ip]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout + 0.5)
        return proc.returncode == 0
    except Exception:
        return False


def build_table(devices: Dict[str, Device]) -> Table:
    table = Table(title="ESP32 ARP Monitor", box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Name")
    table.add_column("IP")
    table.add_column("MAC")
    table.add_column("Status")
    table.add_column("Last Seen")

    for d in devices.values():
        last_seen = d.last_seen.strftime("%H:%M:%S") if d.last_seen else "-"
        status = Text("● Connected", style="bold green") if d.online else Text("● Disconnected", style="bold red")
        table.add_row(d.name, d.ip, d.mac, status, last_seen)

    return table


def main():
    devices: Dict[str, Device] = {}

    with Live(console=console, refresh_per_second=2) as live:
        while True:
            out = run_arp()
            found = parse_arp(out)

            for name, ip, mac in found:
                if mac not in devices:
                    devices[mac] = Device(mac=mac, ip=ip, name=name)
                else:
                    devices[mac].ip = ip
                devices[mac].update_seen()
                devices[mac].update_status()

            live.update(Panel(build_table(devices), border_style="cyan"))
            time.sleep(REFRESH_INTERVAL_S)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
