import cv2
import torch
import pygame
import numpy as np
from ultralytics import YOLO
from collections import deque
import datetime
import csv
import os
import time
import json
import platform
import subprocess
from pathlib import Path

# ─── Constants ────────────────────────────────────────────────────────────────
FPS = 30
WARNING_DURATION = 2
QUEUE_DURATION = 2
YAWN_THRESHOLD_FRAMES = int(FPS * 1)
DROWSY_THRESHOLD_FRAMES = int(FPS * 0.8)
HEAD_THRESHOLD_FRAMES = int(FPS * 0.8)
PHONE_THRESHOLD_FRAMES = int(FPS * 1.0)
ALARM_COOLDOWN = 5
BREAK_REMINDER_MIN = 30
# Always write data files next to the script itself so the dashboard
# can find them with __file__-relative paths too.
_SCRIPT_DIR    = Path(__file__).parent.resolve()
LOG_FILE       = str(_SCRIPT_DIR / "session_log.csv")
SCREENSHOT_DIR = str(_SCRIPT_DIR / "alerts")
PROFILES_FILE  = str(_SCRIPT_DIR / "driver_profiles.json")

# ─── Night Mode ───────────────────────────────────────────────────────────────
NIGHT_MODE_BRIGHTNESS_THRESHOLD = 60   # average pixel value below this → night
CLAHE_CLIP = 3.0
CLAHE_TILE = (8, 8)

# Phone label keywords — any class whose name contains one of these
# will be treated as a phone distraction event. Case-insensitive.
PHONE_KEYWORDS = ["phone", "cell", "mobile", "calling", "texting"]


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-DRIVER PROFILE SUPPORT
# ══════════════════════════════════════════════════════════════════════════════
class DriverProfiles:
    """
    Persists driver profiles in a local JSON file.
    Each profile stores cumulative session statistics.
    """
    def __init__(self, path=PROFILES_FILE):
        self.path = path
        self.profiles = self._load()
        self.active = None          # currently selected profile name

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                return json.load(f)
        return {}

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.profiles, f, indent=2)

    def list_names(self):
        return sorted(self.profiles.keys())

    def select(self, name: str):
        """Select or create a profile by name."""
        if name not in self.profiles:
            self.profiles[name] = {
                "created": datetime.datetime.now().isoformat(),
                "total_sessions": 0,
                "total_drowsy_alerts": 0,
                "total_yawn_alerts": 0,
                "total_head_alerts": 0,
                "total_phone_alerts": 0,
                "total_drive_minutes": 0,
            }
        self.active = name
        print(f"[Profile] Active driver: {name}")

    def update_session(self, stats: dict, duration_minutes: float):
        if not self.active:
            return
        p = self.profiles[self.active]
        p["total_sessions"]      += 1
        p["total_drowsy_alerts"] += stats.get("drowsy_count", 0)
        p["total_yawn_alerts"]   += stats.get("yawn_count", 0)
        p["total_head_alerts"]   += stats.get("head_count", 0)
        p["total_phone_alerts"]  += stats.get("phone_count", 0)
        p["total_drive_minutes"] += round(duration_minutes, 1)
        self.save()

    def get_active_info(self):
        if not self.active or self.active not in self.profiles:
            return {}
        return self.profiles[self.active]

    def delete(self, name: str):
        if name in self.profiles:
            del self.profiles[name]
            self.save()
            if self.active == name:
                self.active = None


# ══════════════════════════════════════════════════════════════════════════════
#  NIGHT MODE / LOW-LIGHT ENHANCEMENT
# ══════════════════════════════════════════════════════════════════════════════
class NightModeProcessor:
    """
    Automatically detects low-light conditions and enhances the frame
    using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Can also be toggled manually.
    """
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
        self.auto = True        # auto-detect
        self.forced = False     # manual override
        self.active = False     # current state

    def toggle_auto(self):
        self.auto = not self.auto

    def toggle_force(self):
        self.forced = not self.forced

    def process(self, frame: np.ndarray):
        """
        Returns (enhanced_frame, is_night_mode_active).
        Enhancement is applied to the frame fed to the model AND the display.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()

        is_dark = brightness < NIGHT_MODE_BRIGHTNESS_THRESHOLD
        self.active = self.forced or (self.auto and is_dark)

        if self.active:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            return enhanced, True, brightness
        return frame, False, brightness


# ══════════════════════════════════════════════════════════════════════════════
#  ENHANCED ALERT SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
class AlertLevel:
    INFO    = "INFO"
    WARNING = "WARNING"
    DANGER  = "DANGER"
    CRITICAL = "CRITICAL"


class AlertBanner:
    """
    Manages a queue of timed on-screen banners with priorities.
    Higher-priority alerts pre-empt lower ones.
    """
    PRIORITY = {AlertLevel.INFO: 0, AlertLevel.WARNING: 1,
                AlertLevel.DANGER: 2, AlertLevel.CRITICAL: 3}
    COLORS   = {
        AlertLevel.INFO:     (100, 200, 100),
        AlertLevel.WARNING:  (0, 200, 255),
        AlertLevel.DANGER:   (0, 100, 255),
        AlertLevel.CRITICAL: (0, 0, 255),
    }

    def __init__(self):
        self.current_level  = None
        self.current_msg    = ""
        self.expiry         = 0.0
        self.flash_state    = False
        self.flash_timer    = 0.0

    def push(self, message: str, level: str, duration: float = 2.5):
        now = time.time()
        if (self.current_level is None
                or self.PRIORITY[level] >= self.PRIORITY.get(self.current_level, 0)
                or now > self.expiry):
            self.current_level = level
            self.current_msg   = message
            self.expiry        = now + duration

    def draw(self, frame: np.ndarray):
        now = time.time()
        if now > self.expiry or not self.current_level:
            self.current_level = None
            return

        # Flash logic for CRITICAL
        if self.current_level == AlertLevel.CRITICAL:
            if now - self.flash_timer > 0.25:
                self.flash_state = not self.flash_state
                self.flash_timer = now
            if not self.flash_state:
                return

        h, w = frame.shape[:2]
        color = self.COLORS[self.current_level]

        # Full-width banner at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), color, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Icon + text
        icon = {"INFO": "ℹ", "WARNING": "⚠", "DANGER": "⚡", "CRITICAL": "🚨"}.get(self.current_level, "!")
        text = f"{icon}  {self.current_msg}"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        x = (w - tw) // 2
        cv2.putText(frame, text, (x, 37),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)

        # Progress bar showing time remaining
        frac = max(0.0, (self.expiry - now) / 2.5)
        cv2.rectangle(frame, (0, 53), (int(w * frac), 55), (255, 255, 255), -1)


class EnhancedAlerts:
    """
    Wraps sound + visual alerts with per-type cooldowns and escalation.
    """
    COOLDOWNS = {
        "drowsy": 4.0,
        "yawn":   6.0,
        "head":   5.0,
        "phone":  5.0,
    }

    def __init__(self, banner: AlertBanner):
        self.banner     = banner
        self.last_times = {k: 0.0 for k in self.COOLDOWNS}
        self._mixer_ready = False

    def _init_mixer(self):
        if not self._mixer_ready:
            pygame.mixer.init()
            self._mixer_ready = True

    def _play(self, sound_file: str, duration_ms: int):
        self._init_mixer()
        if os.path.exists(sound_file):
            sound = pygame.mixer.Sound(sound_file)
            sound.play(loops=0, maxtime=duration_ms)
        else:
            # Fallback: system beep
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(1000, min(duration_ms, 500))
            elif platform.system() == "Darwin":
                subprocess.Popen(["afplay", "/System/Library/Sounds/Ping.aiff"])
            else:
                subprocess.Popen(["paplay", "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga"],
                                 stderr=subprocess.DEVNULL)

    def trigger(self, alert_type: str, score: float):
        now = time.time()
        cooldown = self.COOLDOWNS.get(alert_type, 5.0)
        if now - self.last_times.get(alert_type, 0) < cooldown:
            return
        self.last_times[alert_type] = now

        # Escalate based on drowsiness score
        if score >= 70:
            level, dur_ms = AlertLevel.CRITICAL, 3000
        elif score >= 40:
            level, dur_ms = AlertLevel.DANGER, 2000
        else:
            level, dur_ms = AlertLevel.WARNING, 1000

        messages = {
            "drowsy": "DROWSY DETECTED — Wake up!",
            "yawn":   "YAWN DETECTED — Stay alert!",
            "head":   "HEAD MOVEMENT — Eyes on road!",
            "phone":  "PHONE DISTRACTION — Put it down!",
        }
        self.banner.push(messages.get(alert_type, "ALERT"), level)
        self._play("alarm.wav", dur_ms)

    def push_info(self, msg: str):
        self.banner.push(msg, AlertLevel.INFO, 3.0)


# ══════════════════════════════════════════════════════════════════════════════
#  DROWSINESS SCORE (unchanged logic, minor refactor)
# ══════════════════════════════════════════════════════════════════════════════
class DrowsinessScore:
    def __init__(self):
        self.score = 0.0
        self.last_update = time.time()

    def update(self, drowsy: bool, yawn: bool, head: bool, phone: bool = False):
        now     = time.time()
        elapsed = now - self.last_update
        self.last_update = now
        self.score = max(0.0, self.score - elapsed * 1.5)
        if drowsy: self.score = min(100.0, self.score + 8.0)
        if yawn:   self.score = min(100.0, self.score + 4.0)
        if head:   self.score = min(100.0, self.score + 3.0)
        if phone:  self.score = min(100.0, self.score + 5.0)

    @property
    def level(self):
        if   self.score >= 70: return "CRITICAL", (0, 0, 255)
        elif self.score >= 40: return "WARNING",  (0, 140, 255)
        elif self.score >= 15: return "MILD",     (0, 255, 255)
        else:                  return "ALERT",    (0, 255, 80)


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════════
def init_log():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "event", "drowsiness_score", "driver", "details"])

def log_event(event: str, score: float, driver: str = "", details: str = ""):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="") as f:
        # QUOTE_ALL ensures commas inside 'details' are never
        # misread as column separators (fixes pandas ParserError).
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow([ts, event, f"{score:.1f}", driver, details])

def save_screenshot(frame, reason: str):
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SCREENSHOT_DIR, f"{reason}_{ts}.jpg")
    cv2.imwrite(path, frame)
    print(f"[Screenshot] Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION REPORT EXPORT
# ══════════════════════════════════════════════════════════════════════════════
def export_session_report(stats: dict, score_obj: DrowsinessScore,
                           session_start: datetime.datetime,
                           driver_name: str = "Unknown"):
    """
    Generates a human-readable .txt report and a machine-readable .json
    file for the current session.
    """
    now      = datetime.datetime.now()
    duration = now - session_start
    mins, secs = divmod(int(duration.total_seconds()), 60)

    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)

    ts_str = now.strftime("%Y%m%d_%H%M%S")
    txt_path  = report_dir / f"session_{driver_name}_{ts_str}.txt"
    json_path = report_dir / f"session_{driver_name}_{ts_str}.json"

    # ── Text report ──────────────────────────────────────────────────────────
    separator = "=" * 52
    lines = [
        separator,
        "       DRIVER DROWSINESS SESSION REPORT",
        separator,
        f"  Driver        : {driver_name}",
        f"  Date          : {now.strftime('%Y-%m-%d')}",
        f"  Start Time    : {session_start.strftime('%H:%M:%S')}",
        f"  End Time      : {now.strftime('%H:%M:%S')}",
        f"  Duration      : {mins:02d}m {secs:02d}s",
        separator,
        "  ALERT SUMMARY",
        f"  Drowsy Alerts : {stats['drowsy_count']}",
        f"  Yawn Alerts   : {stats['yawn_count']}",
        f"  Head Alerts   : {stats['head_count']}",
        f"  Phone Alerts  : {stats.get('phone_count', 0)}",
        f"  Final Score   : {score_obj.score:.1f} / 100",
        f"  Risk Level    : {score_obj.level[0]}",
        separator,
        "  SAFETY RATING",
    ]

    total_alerts = (stats["drowsy_count"] + stats["yawn_count"]
                    + stats["head_count"] + stats.get("phone_count", 0))
    if total_alerts == 0:
        rating = "★★★★★  Excellent – No incidents"
    elif total_alerts <= 3:
        rating = "★★★★☆  Good – Minor fatigue signs"
    elif total_alerts <= 8:
        rating = "★★★☆☆  Fair – Consider a break"
    elif total_alerts <= 15:
        rating = "★★☆☆☆  Poor – High fatigue risk"
    else:
        rating = "★☆☆☆☆  Critical – Do not drive!"

    lines += [f"  {rating}", separator, ""]

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ── JSON report ──────────────────────────────────────────────────────────
    report_data = {
        "driver": driver_name,
        "date": now.strftime("%Y-%m-%d"),
        "start_time": session_start.isoformat(),
        "end_time": now.isoformat(),
        "duration_seconds": int(duration.total_seconds()),
        "alerts": {
            "drowsy": stats["drowsy_count"],
            "yawn":   stats["yawn_count"],
            "head":   stats["head_count"],
            "phone":  stats.get("phone_count", 0),
        },
        "final_score": round(score_obj.score, 1),
        "risk_level": score_obj.level[0],
        "safety_rating": rating,
    }
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n📊 Session report saved:")
    print(f"   Text : {txt_path}")
    print(f"   JSON : {json_path}")
    return str(txt_path), str(json_path)


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD OVERLAY
# ══════════════════════════════════════════════════════════════════════════════
def draw_dashboard(frame, score_obj, stats: dict,
                   session_start: datetime.datetime,
                   driver_name: str,
                   night_active: bool,
                   brightness: float):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    panel_x = w - 270
    cv2.rectangle(overlay, (panel_x, 0), (w, h), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    level_text, level_color = score_obj.level

    # Title
    cv2.putText(frame, "DROWSINESS", (panel_x + 20, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, "MONITOR", (panel_x + 30, 52),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 200, 200), 1)
    cv2.line(frame, (panel_x + 10, 62), (w - 10, 62), (60, 60, 80), 1)

    # Driver name
    cv2.putText(frame, f"Driver: {driver_name[:14]}", (panel_x + 10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 200, 255), 1)

    # Fatigue score bar
    cv2.putText(frame, "FATIGUE SCORE", (panel_x + 10, 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
    bar_w = int((score_obj.score / 100.0) * 230)
    cv2.rectangle(frame, (panel_x + 10, 105), (panel_x + 240, 123), (40, 40, 50), -1)
    cv2.rectangle(frame, (panel_x + 10, 105), (panel_x + 10 + bar_w, 123), level_color, -1)
    cv2.putText(frame, f"{score_obj.score:.0f}/100", (panel_x + 85, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    cv2.putText(frame, level_text, (panel_x + 70, 147),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, level_color, 2)
    cv2.line(frame, (panel_x + 10, 157), (w - 10, 157), (60, 60, 80), 1)

    # Stats
    session_dur = datetime.datetime.now() - session_start
    mins, secs  = divmod(int(session_dur.total_seconds()), 60)

    stats_lines = [
        ("Session",       f"{mins:02d}:{secs:02d}"),
        ("Drowsy Alerts", str(stats["drowsy_count"])),
        ("Yawn Alerts",   str(stats["yawn_count"])),
        ("Head Alerts",   str(stats["head_count"])),
        ("Phone Alerts",  str(stats.get("phone_count", 0))),
        ("FPS",           str(stats.get("fps", "--"))),
    ]
    y = 182
    for label, val in stats_lines:
        cv2.putText(frame, label, (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (140, 140, 150), 1)
        cv2.putText(frame, val, (panel_x + 165, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
        y += 22

    cv2.line(frame, (panel_x + 10, y), (w - 10, y), (60, 60, 80), 1)
    y += 15

    # Break reminder
    if mins >= BREAK_REMINDER_MIN:
        cv2.putText(frame, "! TAKE A BREAK !", (panel_x + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 100, 255), 1)
    else:
        cv2.putText(frame, f"Break in {BREAK_REMINDER_MIN - mins} min",
                    (panel_x + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 180, 100), 1)
    y += 20

    # Night mode indicator
    nm_color = (0, 200, 255) if night_active else (70, 70, 80)
    nm_label = f"Night Mode {'ON' if night_active else 'OFF'}  [{brightness:.0f}lx]"
    cv2.putText(frame, nm_label, (panel_x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, nm_color, 1)

    # Key hints
    hints = ["Q:Quit  N:Night  R:Report", "P:Profile  D:Del Profile"]
    cv2.putText(frame, hints[0], (panel_x + 5, h - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 80, 90), 1)
    cv2.putText(frame, hints[1], (panel_x + 5, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 80, 90), 1)


# ══════════════════════════════════════════════════════════════════════════════
#  PROFILE SELECTION CLI
# ══════════════════════════════════════════════════════════════════════════════
def select_profile_cli(profiles: DriverProfiles) -> str:
    print("\n" + "="*50)
    print("  DRIVER PROFILE SELECTION")
    print("="*50)
    existing = profiles.list_names()
    if existing:
        print("  Existing profiles:")
        for i, name in enumerate(existing, 1):
            p = profiles.profiles[name]
            print(f"    [{i}] {name}  "
                  f"(sessions: {p['total_sessions']}, "
                  f"drive time: {p['total_drive_minutes']:.0f} min)")
    print("  [N] New profile")
    print("="*50)
    choice = input("  Select number or enter name: ").strip()
    if choice.upper() == "N" or not choice:
        name = input("  Enter new driver name: ").strip() or "Driver1"
    elif choice.isdigit() and 1 <= int(choice) <= len(existing):
        name = existing[int(choice) - 1]
    else:
        name = choice
    profiles.select(name)
    return name


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def get_webcam_fps():
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30
    cap.release()
    return fps if fps > 0 else 30

def load_model(model_path):
    return YOLO(model_path)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN DETECTION LOOP
# ══════════════════════════════════════════════════════════════════════════════
def webcam_detection(model, fps, profiles: DriverProfiles):
    queue_len    = int(fps * QUEUE_DURATION)
    eye_queue    = deque(maxlen=queue_len)
    yawn_queue   = deque(maxlen=queue_len)
    head_queue   = deque(maxlen=queue_len)
    phone_queue  = deque(maxlen=queue_len)
    frame_times  = deque(maxlen=30)

    score_obj     = DrowsinessScore()
    night_proc    = NightModeProcessor()
    banner        = AlertBanner()
    alerts        = EnhancedAlerts(banner)
    session_start = datetime.datetime.now()
    driver_name   = profiles.active or "Unknown"

    stats = {"drowsy_count": 0, "yawn_count": 0,
             "head_count": 0, "phone_count": 0, "fps": int(fps)}

    init_log()
    log_event("SESSION_START", 0, driver=driver_name, details=f"FPS={fps}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access webcam.")
        return

    print("✅ Detection started.")
    print("   Q=Quit | N=Toggle Night Mode | R=Export Report | P=Profile info | D=Delete Profile")

    night_active = False
    brightness   = 128.0

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # ── Night mode pre-processing ────────────────────────────────────────
        processed_frame, night_active, brightness = night_proc.process(frame)

        # ── Inference ────────────────────────────────────────────────────────
        img     = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        results = model.predict(source=[img], save=False, verbose=False)[0]

        cur_eye   = False
        cur_yawn  = False
        cur_head  = False
        cur_phone = False

        # Draw boxes on the (possibly enhanced) display frame
        display_frame = processed_frame.copy()

        for result in results:
            boxes   = result.boxes
            xyxy    = boxes.xyxy.cpu().numpy()
            confs   = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            for i in range(len(xyxy)):
                xmin, ymin, xmax, ymax = map(int, xyxy[i])
                conf  = confs[i]
                label = int(classes[i])
                if conf < 0.5:
                    continue
                color = (0, 255, 0)
                name  = model.names[label]

                if label in [0, 1, 2]:
                    cur_eye = True
                if label in [4, 5]:
                    cur_head = True; color = (0, 255, 255)
                if label == 8:
                    cur_yawn = True; color = (0, 200, 255)
                if any(kw in name.lower() for kw in PHONE_KEYWORDS):
                    cur_phone = True; color = (255, 50, 50)

                cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(display_frame, f"{name} {conf:.2f}",
                            (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # ── Queue updates ────────────────────────────────────────────────────
        eye_queue.append(cur_eye)
        yawn_queue.append(cur_yawn)
        head_queue.append(cur_head)
        phone_queue.append(cur_phone)

        eye_count   = sum(eye_queue)
        yawn_count  = sum(yawn_queue)
        head_count  = sum(head_queue)
        phone_count = sum(phone_queue)

        dthresh = int(fps * 0.8)
        ythresh = int(fps * 1.0)
        hthresh = int(fps * 0.8)
        pthresh = int(fps * 1.0)

        events = set()

        if eye_count >= dthresh:
            events.add("drowsy")
            stats["drowsy_count"] += 1
            alerts.trigger("drowsy", score_obj.score)
            save_screenshot(display_frame, "drowsy")
            log_event("DROWSY", score_obj.score, driver_name)

        if yawn_count >= ythresh:
            events.add("yawn")
            stats["yawn_count"] += 1
            alerts.trigger("yawn", score_obj.score)
            log_event("YAWN", score_obj.score, driver_name)
            yawn_queue.clear()

        if head_count >= hthresh:
            events.add("head")
            stats["head_count"] += 1
            alerts.trigger("head", score_obj.score)
            log_event("HEAD_MOVEMENT", score_obj.score, driver_name)
            head_queue.clear()

        if phone_count >= pthresh:
            events.add("phone")
            stats["phone_count"] += 1
            alerts.trigger("phone", score_obj.score)
            save_screenshot(display_frame, "phone")
            log_event("PHONE_DISTRACTION", score_obj.score, driver_name)
            phone_queue.clear()

        # ── Update score ─────────────────────────────────────────────────────
        score_obj.update(
            drowsy="drowsy" in events,
            yawn="yawn"     in events,
            head="head"     in events,
            phone="phone"   in events,
        )

        # Re-colour eye boxes
        for result in results:
            boxes   = result.boxes
            xyxy    = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            for i in range(len(xyxy)):
                xmin, ymin, xmax, ymax = map(int, xyxy[i])
                if int(classes[i]) in [0, 1, 2]:
                    c = (0, 0, 255) if "drowsy" in events else (0, 255, 0)
                    cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), c, 2)
                    cv2.putText(display_frame, model.names[int(classes[i])],
                                (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

        # ── Live FPS ─────────────────────────────────────────────────────────
        frame_times.append(time.time() - t0)
        stats["fps"] = int(1.0 / (sum(frame_times) / len(frame_times)))

        # ── Draw dashboard ───────────────────────────────────────────────────
        draw_dashboard(display_frame, score_obj, stats, session_start,
                       driver_name, night_active, brightness)

        # ── Draw alert banner (on top of everything) ─────────────────────────
        banner.draw(display_frame)

        cv2.imshow("Driver Drowsiness Detection", display_frame)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            night_proc.toggle_force()
            state = "ON (forced)" if night_proc.forced else "OFF (auto)"
            alerts.push_info(f"Night Mode: {state}")
        elif key == ord("r"):
            txt, jsn = export_session_report(
                stats, score_obj, session_start, driver_name)
            alerts.push_info("Report exported! Check /reports folder.")
        elif key == ord("p"):
            info = profiles.get_active_info()
            if info:
                msg = (f"Sessions:{info['total_sessions']} "
                       f"Drowsy:{info['total_drowsy_alerts']} "
                       f"Drive:{info['total_drive_minutes']:.0f}min")
                print(f"\n[Profile: {driver_name}] {msg}")
                alerts.push_info(f"Profile: {msg[:40]}")
        elif key == ord("d"):
            confirm = input(f"\nDelete profile '{driver_name}'? (y/n): ").strip()
            if confirm.lower() == "y":
                profiles.delete(driver_name)
                print(f"[Profile] '{driver_name}' deleted.")
                alerts.push_info(f"Profile '{driver_name}' deleted.")

    # ── Session end ───────────────────────────────────────────────────────────
    duration_min = (datetime.datetime.now() - session_start).total_seconds() / 60
    profiles.update_session(stats, duration_min)

    log_event("SESSION_END", score_obj.score, driver_name,
              f"drowsy={stats['drowsy_count']},yawn={stats['yawn_count']},"
              f"head={stats['head_count']},phone={stats.get('phone_count',0)}")

    export_session_report(stats, score_obj, session_start, driver_name)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n📋 Session log: {LOG_FILE}")
    print(f"📸 Screenshots: {SCREENSHOT_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fps   = get_webcam_fps()
    print(f"웹캠 프레임 수: {fps} FPS")

    model = load_model("best.pt")

    # ── Print full class list so you can verify phone label detection ──────────
    print("\n" + "="*55)
    print("  MODEL CLASS LIST")
    print("="*55)
    phone_found = []
    for idx, name in model.names.items():
        matched = any(kw in name.lower() for kw in PHONE_KEYWORDS)
        tag = "  ← 📱 PHONE DETECTED" if matched else ""
        print(f"  [{idx:>2}] {name}{tag}")
        if matched:
            phone_found.append((idx, name))
    print("="*55)
    if phone_found:
        print(f"  ✅ Phone classes matched: {phone_found}")
    else:
        print("  ⚠  No phone class found in this model.")
        print(f"     Keywords checked: {PHONE_KEYWORDS}")
        print("     Phone alerts will be disabled for this session.")
        print("     Add the correct class name keyword to PHONE_KEYWORDS if needed.")
    print("="*55 + "\n")

    profiles = DriverProfiles()
    select_profile_cli(profiles)

    webcam_detection(model, fps, profiles)