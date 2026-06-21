"""Generate a simple 512x512 RGBA app icon (no third-party deps).

Draws a rounded-square purple tile with a white microphone glyph, which the
Tauri CLI (`npx tauri icon`) then expands into every required icon format.
"""
import struct
import zlib
import math

SIZE = 512
# Colors (RGBA)
BG_TOP = (124, 92, 231)     # #7C5CE7
BG_BOT = (90, 70, 200)
MIC = (255, 255, 255)


def rounded_alpha(x, y, size, radius):
    """Anti-aliased alpha for a rounded square covering the canvas."""
    cx = min(x, size - 1 - x)
    cy = min(y, size - 1 - y)
    if cx >= radius or cy >= radius:
        return 255
    dx = radius - cx
    dy = radius - cy
    if dx <= 0 or dy <= 0:
        return 255
    dist = math.hypot(dx, dy)
    if dist <= radius - 1:
        return 255
    if dist >= radius + 1:
        return 0
    return int(255 * (radius + 1 - dist) / 2)


def in_mic(x, y, size):
    """Rough microphone shape: capsule body + stand + base."""
    cx = size / 2
    # Capsule body
    body_w = size * 0.16
    body_top = size * 0.24
    body_bot = size * 0.56
    if abs(x - cx) <= body_w:
        if body_top <= y <= body_bot:
            # rounded top/bottom of the capsule
            if y < body_top + body_w:
                if math.hypot(x - cx, y - (body_top + body_w)) <= body_w:
                    return True
            elif y > body_bot - body_w:
                if math.hypot(x - cx, y - (body_bot - body_w)) <= body_w:
                    return True
            else:
                return True
    # Arc / cradle (open ring under the capsule)
    r = size * 0.20
    d = math.hypot(x - cx, y - body_bot)
    if r - size * 0.03 <= d <= r and y >= body_bot:
        return True
    # Stand
    if abs(x - cx) <= size * 0.018 and body_bot <= y <= size * 0.80:
        return True
    # Base
    if abs(x - cx) <= size * 0.12 and size * 0.78 <= y <= size * 0.81:
        return True
    return False


def build():
    raw = bytearray()
    radius = int(SIZE * 0.22)
    for y in range(SIZE):
        raw.append(0)  # PNG filter type 0 for this scanline
        t = y / (SIZE - 1)
        bg = tuple(int(BG_TOP[i] + (BG_BOT[i] - BG_TOP[i]) * t) for i in range(3))
        for x in range(SIZE):
            a = rounded_alpha(x, y, SIZE, radius)
            if a == 0:
                raw.extend((0, 0, 0, 0))
                continue
            if in_mic(x, y, SIZE):
                raw.extend((MIC[0], MIC[1], MIC[2], a))
            else:
                raw.extend((bg[0], bg[1], bg[2], a))
    return bytes(raw)


def chunk(tag, data):
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def main():
    raw = build()
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", SIZE, SIZE, 8, 6, 0, 0, 0)  # 8-bit RGBA
    png = (
        sig
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    with open("app-icon.png", "wb") as f:
        f.write(png)
    print("wrote app-icon.png", len(png), "bytes")


if __name__ == "__main__":
    main()
