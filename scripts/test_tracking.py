"""Phase 5 tracking and identity tests.

Run from the project root:

    python scripts/test_tracking.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vision.recognition.catalog import load_catalog
from vision.recognition.identity import ProductIdentifier
from vision.tracking.manager import TrackManager
from vision.tracking.tracks import TrackState

PASS = 0
FAIL = 0
IDENTIFIER = ProductIdentifier(use_database=False)

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        extra = f" — {detail}" if detail else ""
        print(f"  FAIL  {name}{extra}")


def _det(class_id: int, bbox: list[float], confidence: float = 0.9) -> dict:
    return IDENTIFIER.identify(
        {"class_id": class_id, "confidence": confidence, "bbox": bbox}
    )


def test_identity_mapping() -> None:
    print("Identity mapping")
    catalog = load_catalog()
    identifier = ProductIdentifier(catalog=catalog, use_database=False)
    coke = identifier.identify({"class_id": 0, "confidence": 0.94, "bbox": [100, 150, 300, 500]})
    check("class 0 → yaml product_id 101", coke["product_id"] == 101)
    check("class 0 → sku COKE500", coke["sku"] == "COKE500")
    check("class 0 → Coca-Cola 500ml", coke["product_name"] == "Coca-Cola 500ml")
    check("yaml identity has no price", coke["price"] is None)
    maggi = identifier.identity_for_class(2)
    check("class 2 → yaml product_id 103", maggi.product_id == 103)
    check("8 catalog SKUs", len(catalog.products) == 8)


def test_duplicate_prevention() -> None:
    print("Duplicate prevention (100 frames, one Coke)")
    manager = TrackManager(stable_frames=3, max_missing=10)
    bbox = [40.0, 40.0, 140.0, 220.0]
    last_tracks = []
    for _frame in range(100):
        last_tracks = manager.update([_det(0, bbox, 0.91)])
        bbox = [bbox[0] + 0.4, bbox[1] + 0.1, bbox[2] + 0.4, bbox[3] + 0.1]
    check("one track across 100 frames", len(last_tracks) == 1, str(len(last_tracks)))
    check("same track_id", last_tracks[0]["track_id"] == 1)
    check("cart Coke × 1", manager.cart_by_name() == {"Coca-Cola 500ml": 1}, str(manager.cart_by_name()))
    public = manager.visible_outputs()
    required = {
        "track_id",
        "product_id",
        "class_id",
        "product_name",
        "confidence",
        "bbox",
    }
    check("public JSON has contract fields", required.issubset(public[0]), str(sorted(public[0])))


def test_two_same_sku() -> None:
    print("Two physical Cokes")
    manager = TrackManager(stable_frames=2, max_missing=8)
    for _frame in range(12):
        manager.update(
            [
                _det(0, [20, 20, 80, 140], 0.9),
                _det(0, [200, 20, 260, 140], 0.88),
            ]
        )
    ids = sorted(track.track_id for track in manager.confirmed_tracks())
    check("two track ids", ids == [1, 2], str(ids))
    check("Coke × 2", manager.cart_by_name() == {"Coca-Cola 500ml": 2}, str(manager.cart_by_name()))


def test_five_products_simultaneous() -> None:
    print("Five products simultaneously (100 frames)")
    manager = TrackManager(stable_frames=4, max_missing=12)
    # class_id 0-4: Coke, Lays, Maggi, Dairy Milk, Pepsi
    bases = [
        [30.0, 40.0, 110.0, 180.0],
        [150.0, 50.0, 240.0, 160.0],
        [270.0, 30.0, 360.0, 170.0],
        [400.0, 60.0, 490.0, 150.0],
        [520.0, 40.0, 610.0, 190.0],
    ]
    for frame in range(100):
        detections = []
        for class_id, box in enumerate(bases):
            dx = (frame % 7) * 0.8
            dy = (frame % 5) * 0.4
            moved = [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]
            detections.append(_det(class_id, moved, 0.85 + class_id * 0.01))
        tracks = manager.update(detections)

    check("five active tracks", len(manager.confirmed_tracks()) == 5, str(len(manager.confirmed_tracks())))
    check("five unique ids", len({t.track_id for t in manager.confirmed_tracks()}) == 5)
    cart = manager.cart_by_name()
    expected = {
        "Coca-Cola 500ml": 1,
        "Lays Classic": 1,
        "Maggi Noodles": 1,
        "Dairy Milk": 1,
        "Pepsi 500ml": 1,
    }
    check("cart is ×1 each, not ×100", cart == expected, str(cart))
    check("no sixth product", sum(cart.values()) == 5)
    sample = manager.visible_outputs()
    check("five public outputs", len(sample) == 5, str(len(sample)))
    print("  sample:", json.dumps(sample[0], separators=(",", ":")))


def test_entry_exit() -> None:
    print("Entry / remain / leave")
    manager = TrackManager(stable_frames=3, max_missing=4)
    box = [10.0, 10.0, 90.0, 120.0]
    states = []
    for frame in range(14):
        if frame < 8:
            tracks = manager.update([_det(1, box, 0.9)])
        else:
            tracks = manager.update([])
        if tracks:
            states.append(tracks[0]["state"])
        else:
            states.append(TrackState.EXITED.value)
    check("starts entering", states[0] == TrackState.ENTERING.value, states[0])
    check("becomes visible after stable frames", TrackState.VISIBLE.value in states, str(states))
    check("leaving after misses", TrackState.LEAVING.value in states, str(states))
    last = manager.tracks[1]
    check("exited after max_missing", last.state == TrackState.EXITED, last.state.value)
    check("exited item still in session cart", manager.cart_by_name() == {"Lays Classic": 1})


def test_confidence_stabilization() -> None:
    print("Confidence stabilization")
    manager = TrackManager(stable_frames=5, max_missing=8)
    box = [50.0, 50.0, 120.0, 160.0]
    for _frame in range(4):
        manager.update([_det(6, box, 0.55)])
    check("not in cart before 5 frames", manager.cart_by_name() == {}, str(manager.cart_by_name()))
    check("state entering", manager.tracks[1].state == TrackState.ENTERING)
    manager.update([_det(6, box, 0.62)])
    check("confirmed on 5th consistent frame", manager.tracks[1].confirmed)
    check("KitKat × 1 after stable", manager.cart_by_name() == {"KitKat": 1})


def test_weak_flicker_rejected() -> None:
    print("Weak flicker rejected")
    manager = TrackManager(stable_frames=5, max_missing=3)
    box = [80.0, 80.0, 140.0, 180.0]
    for frame in range(20):
        if frame % 3 == 0:
            manager.update([_det(3, box, 0.4)])
        else:
            manager.update([])
    check("flicker never reaches cart", manager.cart_track_ids == set(), str(manager.cart_track_ids))


def main() -> int:
    print("Retail Vision — Phase 5 tests\n")
    test_identity_mapping()
    test_duplicate_prevention()
    test_two_same_sku()
    test_five_products_simultaneous()
    test_entry_exit()
    test_confidence_stabilization()
    test_weak_flicker_rejected()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
