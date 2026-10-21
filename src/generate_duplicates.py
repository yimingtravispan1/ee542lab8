from pathlib import Path

from scapy.all import rdpcap, wrpcap


ROOT = Path(__file__).resolve().parents[1]

packets = list(rdpcap(str(ROOT / "data" / "images_shuffled.pcap")))

duplicates = [packet.copy() for packet in packets[:100]]
packets.extend(duplicates)

for i, packet in enumerate(packets):
    packet.time = 1700000000 + i * 0.000001

output = ROOT / "data" / "images_duplicates.pcap"
wrpcap(str(output), packets)

print("Total packets:", len(packets))
print("Added duplicates:", len(duplicates))
print("Saved:", output)