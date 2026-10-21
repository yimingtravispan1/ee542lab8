import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    original = np.load(ROOT / "data" / "original_frames.npy")
    expected_bytes = original.tobytes()
    reports = []
    all_passed = True

    for mode in ("ordered", "shuffled"):
        path = ROOT / "data" / f"frames_rx_{mode}.bin"
        received_bytes = path.read_bytes()

        report = {
            "mode": mode,
            "expected_bytes": len(expected_bytes),
            "received_bytes": len(received_bytes),
            "original_sha256": sha256(expected_bytes),
            "received_sha256": sha256(received_bytes),
            "byte_identical": received_bytes == expected_bytes,
        }

        print(f"\n=== {mode} ===")
        print("Expected bytes:", len(expected_bytes))
        print("Received bytes:", len(received_bytes))

        if len(received_bytes) == len(expected_bytes):
            received = np.frombuffer(
                received_bytes, dtype=np.uint8
            ).reshape(original.shape)

            matches = [
                bool(np.array_equal(original[i], received[i]))
                for i in range(len(original))
            ]
            report["per_frame_match"] = matches

            for frame_id, matched in enumerate(matches):
                print(f"Frame {frame_id:02d}: {'PASS' if matched else 'FAIL'}")
        else:
            print("FAIL: output size does not match")

        print("Byte-for-byte identical:", report["byte_identical"])
        print("Original SHA256:", report["original_sha256"])
        print("Received SHA256:", report["received_sha256"])

        reports.append(report)
        all_passed = all_passed and report["byte_identical"]

    output = ROOT / "results" / "part4_verification.json"
    output.write_text(
        json.dumps(reports, indent=2),
        encoding="utf-8",
    )

    if not all_passed:
        raise SystemExit("Verification FAILED")

    print("\nAll ordered and shuffled frames match exactly.")


if __name__ == "__main__":
    main()