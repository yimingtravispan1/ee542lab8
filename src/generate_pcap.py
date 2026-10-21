import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image
from scapy.all import Ether, IP, UDP, Raw, wrpcap


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

HEIGHT = 224
WIDTH = 224
CHANNELS = 3
NUM_FRAMES = 16
CHUNK_BYTES = 1400
MAGIC = 0x45453534
HEADER = struct.Struct("!IIHHIHI")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle packet arrival order",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(542)
    images = rng.integers(
        0, 256,
        size=(NUM_FRAMES, HEIGHT, WIDTH, CHANNELS),
        dtype=np.uint8,
    )

    np.save(DATA_DIR / "original_frames.npy", images)
    (DATA_DIR / "original_frames.bin").write_bytes(images.tobytes())
    for frame_id, image in enumerate(images):
        Image.fromarray(image).save(
            DATA_DIR / f"original_{frame_id:02d}.png"
        )

    frame_bytes = HEIGHT * WIDTH * CHANNELS
    nseg = math.ceil(frame_bytes / CHUNK_BYTES)
    packets = []

    for frame_id, image in enumerate(images):
        raw = image.tobytes()

        for seq in range(nseg):
            offset = seq * CHUNK_BYTES
            chunk = raw[offset:offset + CHUNK_BYTES]

            header = HEADER.pack(
                MAGIC,
                frame_id,
                seq,
                nseg,
                offset,
                len(chunk),
                frame_bytes,
            )

            packet = (
                Ether(
                    src="02:00:00:00:00:01",
                    dst="02:00:00:00:00:02",
                )
                / IP(src="10.0.0.1", dst="10.0.0.2")
                / UDP(sport=40000, dport=5000)
                / Raw(load=header + chunk)
            )
            packets.append(packet)

    max_ip_bytes = max(len(bytes(packet[IP])) for packet in packets)
    if max_ip_bytes > 1500:
        raise RuntimeError(f"IP packet exceeds MTU: {max_ip_bytes}")

    if args.shuffle:
        order = np.random.default_rng(543).permutation(len(packets))
        packets = [packets[int(i)] for i in order]
    for i, packet in enumerate(packets):
        packet.time = 1700000000 + i * 0.000001

    filename = "images_shuffled.pcap" if args.shuffle else "images.pcap"
    output_path = DATA_DIR / filename
    wrpcap(str(output_path), packets)

    metadata = {
        "frames": NUM_FRAMES,
        "height": HEIGHT,
        "width": WIDTH,
        "channels": CHANNELS,
        "layout": "HWC",
        "dtype": "uint8",
        "frame_bytes": frame_bytes,
        "chunk_bytes": CHUNK_BYTES,
        "header_bytes": HEADER.size,
        "packets_per_frame": nseg,
        "total_packets": len(packets),
        "max_ip_packet_bytes": max_ip_bytes,
        "magic_hex": hex(MAGIC),
        "header_format": "!IIHHIHI",
        "udp_destination_port": 5000,
        "shuffled": args.shuffle,
    }

    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()