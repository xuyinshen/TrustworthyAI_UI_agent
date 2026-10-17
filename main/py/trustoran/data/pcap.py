"""A dependency-free pcap reader + flow-feature extractor.

The O-RAN DDoS dataset ships raw ``.pcap`` files. Rather than pull in scapy /
dpkt (heavy, sometimes needs a C toolchain), we parse the classic libpcap
container directly with ``struct`` and decode Ethernet -> IPv4 -> TCP/UDP just
far enough to compute the flow-level features a Near-RT-RIC xApp would use
(packet rate, size distribution, SYN ratio, protocol mix, ...).

Only what we need is decoded; malformed packets are skipped defensively.
"""
from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PCAP_MAGIC_LE = 0xA1B2C3D4
PCAP_MAGIC_BE = 0xD4C3B2A1
PCAP_MAGIC_NS_LE = 0xA1B23C4D
PCAP_MAGIC_NS_BE = 0x4D3CB2A1

LINKTYPE_ETHERNET = 1
LINKTYPE_RAW = 101
ETH_P_IP = 0x0800
IPPROTO_TCP = 6
IPPROTO_UDP = 17
TCP_FIN, TCP_SYN, TCP_RST, TCP_PSH, TCP_ACK = 0x01, 0x02, 0x04, 0x08, 0x10


@dataclass
class Packet:
    ts: float                 # seconds (float)
    length: int               # on-wire frame length
    proto: str                # "tcp" | "udp" | "other"
    flags: int = 0            # TCP flags (0 for non-TCP)
    src_ip: str = ""
    dst_ip: str = ""
    src_port: int = 0
    dst_port: int = 0


def read_pcap(path) -> List[Packet]:
    """Parse a libpcap file into a list of :class:`Packet` (best effort)."""
    path = Path(path)
    with open(path, "rb") as fh:
        header = fh.read(24)
        if len(header) < 24:
            return []
        magic = struct.unpack("<I", header[:4])[0]
        if magic in (PCAP_MAGIC_LE, PCAP_MAGIC_NS_LE):
            endian, nano = "<", magic == PCAP_MAGIC_NS_LE
        elif magic in (PCAP_MAGIC_BE, PCAP_MAGIC_NS_BE):
            endian, nano = ">", magic == PCAP_MAGIC_NS_BE
        else:
            return []  # not a classic pcap (could be pcapng) -> bail out
        linktype = struct.unpack(endian + "I", header[20:24])[0]

        packets: List[Packet] = []
        rec_hdr = struct.Struct(endian + "IIII")
        while True:
            rh = fh.read(16)
            if len(rh) < 16:
                break
            ts_sec, ts_frac, incl_len, _orig_len = rec_hdr.unpack(rh)
            data = fh.read(incl_len)
            if len(data) < incl_len:
                break
            ts = ts_sec + (ts_frac / 1e9 if nano else ts_frac / 1e6)
            pkt = _decode(data, linktype, ts, incl_len)
            if pkt is not None:
                packets.append(pkt)
        return packets


def _decode(data: bytes, linktype: int, ts: float, length: int) -> Optional[Packet]:
    if linktype == LINKTYPE_ETHERNET:
        if len(data) < 14:
            return None
        ethertype = struct.unpack(">H", data[12:14])[0]
        if ethertype != ETH_P_IP:
            return Packet(ts=ts, length=length, proto="other")
        ip = data[14:]
    elif linktype == LINKTYPE_RAW:
        ip = data
    else:
        return Packet(ts=ts, length=length, proto="other")

    if len(ip) < 20:
        return None
    ihl = (ip[0] & 0x0F) * 4
    if len(ip) < ihl:
        return None
    proto_num = ip[9]
    src_ip = ".".join(str(b) for b in ip[12:16])
    dst_ip = ".".join(str(b) for b in ip[16:20])
    l4 = ip[ihl:]

    if proto_num == IPPROTO_TCP and len(l4) >= 20:
        src_port, dst_port = struct.unpack(">HH", l4[0:4])
        flags = l4[13]
        return Packet(ts, length, "tcp", flags, src_ip, dst_ip, src_port, dst_port)
    if proto_num == IPPROTO_UDP and len(l4) >= 8:
        src_port, dst_port = struct.unpack(">HH", l4[0:4])
        return Packet(ts, length, "udp", 0, src_ip, dst_ip, src_port, dst_port)
    return Packet(ts, length, "other", 0, src_ip, dst_ip, 0, 0)


@dataclass
class FlowFeatures:
    """Aggregated flow-level features for one capture file."""
    values: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        return dict(self.values)


HTTP_PORTS = {80, 8080, 8000, 8888, 3128}

#: ordered feature names produced by :func:`extract_flow_features`
FLOW_FEATURE_NAMES = [
    "n_packets", "n_bytes", "duration", "pkt_rate",
    "avg_pkt_len", "std_pkt_len", "min_pkt_len", "max_pkt_len",
    "mean_iat_ms", "std_iat_ms", "max_iat_ms",
    "tcp_ratio", "udp_ratio", "syn_ratio", "synack_ratio", "ack_ratio",
    "rst_ratio", "http_ratio", "n_src_ips", "n_dst_ports", "payload_ratio",
]


def extract_flow_features(path) -> Optional[FlowFeatures]:
    """Decode one pcap and return aggregated flow features (None if unreadable)."""
    packets = read_pcap(path)
    if not packets:
        return None

    n = len(packets)
    lengths = np.array([p.length for p in packets], dtype=float)
    ts = np.array([p.ts for p in packets], dtype=float)
    ts_sorted = np.sort(ts)
    iats = np.diff(ts_sorted) if n > 1 else np.array([0.0])
    duration = float(ts_sorted[-1] - ts_sorted[0]) if n > 1 else 0.0

    tcp = sum(1 for p in packets if p.proto == "tcp")
    udp = sum(1 for p in packets if p.proto == "udp")
    syn = sum(1 for p in packets if p.proto == "tcp" and (p.flags & TCP_SYN) and not (p.flags & TCP_ACK))
    synack = sum(1 for p in packets if p.proto == "tcp" and (p.flags & TCP_SYN) and (p.flags & TCP_ACK))
    ack = sum(1 for p in packets if p.proto == "tcp" and (p.flags & TCP_ACK))
    rst = sum(1 for p in packets if p.proto == "tcp" and (p.flags & TCP_RST))
    http = sum(
        1 for p in packets
        if p.src_port in HTTP_PORTS or p.dst_port in HTTP_PORTS
    )
    # payload bytes approximation: TCP frames beyond the 40-byte eth+ip+tcp header
    payload_bytes = sum(max(0, p.length - 54) for p in packets if p.proto == "tcp")
    total_bytes = float(lengths.sum())

    src_ips = Counter(p.src_ip for p in packets if p.src_ip)
    dst_ports = Counter(p.dst_port for p in packets if p.dst_port)

    vals = {
        "n_packets": float(n),
        "n_bytes": total_bytes,
        "duration": duration,
        "pkt_rate": n / duration if duration > 0 else float(n),
        "avg_pkt_len": float(lengths.mean()),
        "std_pkt_len": float(lengths.std()),
        "min_pkt_len": float(lengths.min()),
        "max_pkt_len": float(lengths.max()),
        "mean_iat_ms": float(iats.mean() * 1000.0),
        "std_iat_ms": float(iats.std() * 1000.0),
        "max_iat_ms": float(iats.max() * 1000.0),
        "tcp_ratio": tcp / n,
        "udp_ratio": udp / n,
        "syn_ratio": syn / n,
        "synack_ratio": synack / n,
        "ack_ratio": ack / n,
        "rst_ratio": rst / n,
        "http_ratio": http / n,
        "n_src_ips": float(len(src_ips)),
        "n_dst_ports": float(len(dst_ports)),
        "payload_ratio": payload_bytes / total_bytes if total_bytes > 0 else 0.0,
    }
    return FlowFeatures(values=vals)
