# Reachy GPT App — Claude Code Reference

## Project overview

Connect a **Reachy Mini** robot (Pollen Robotics) to **OpenAI GPT-4o** for
natural conversation, with:

- Real-time **face tracking** — neck (yaw/pitch) and mobile-base body rotation
- **Face recognition** — remembers specific people and animals by name
- **Persistent memory** via SQLite — last 10 messages + GPT-generated summary per person

## Repository layout

```
reachy-gpt-app/
├── main.py                          # Entry point; orchestrates all threads
├── modules/
│   ├── __init__.py
│   ├── conversation.py              # GPT-4o chat manager
│   ├── face_tracking.py             # OpenCV Haar cascade + neck control
│   ├── face_recognition_module.py   # face_recognition lib + pickle storage
│   └── memory.py                    # SQLite: persons / conversations / summaries
├── requirements.txt
├── .env.template                    # Copy → .env and fill values
├── CLAUDE.md                        # This file
└── README.md
```

## Environment variables (`.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | — | OpenAI API key |
| `REACHY_IP` | robot mode | — | IP address of Reachy Mini |
| `CAMERA_INDEX` | no | `0` | OpenCV camera device index |
| `FACE_CONFIDENCE_THRESHOLD` | no | `0.5` | Lower = stricter face recognition |
| `ENCODINGS_PATH` | no | `face_encodings.pkl` | Path to pickle file |
| `DB_PATH` | no | `reachy_memory.db` | SQLite database path |

## Thread architecture (`main.py`)

```
camera_loop        (daemon)  → captures frames at ~30 fps
tracking_loop      (daemon)  → moves neck at 20 Hz using latest face position
recognition_loop   (daemon)  → identifies person every 15 frames
conversation_loop  (main)    → stdin → GPT-4o → stdout + DB persistence
```

All threads share `SharedState` which uses `threading.Lock` for safe access.
`stop_event` (a `threading.Event`) signals all threads to exit cleanly.

## Module contracts

### `modules/memory.py`
- `init_db()` — create tables (idempotent)
- `get_or_create_person(name) -> int` — returns `person_id`
- `save_message(person_id, role, content) -> int` — persists one turn; triggers auto-summarize at 50 messages
- `load_recent_messages(person_id, limit=10) -> list[dict]` — `[{role, content}, ...]`
- `build_memory_context(person_id) -> str` — returns string for system prompt injection

### `modules/conversation.py`
- `ConversationManager.set_person(name, memory_context)` — resets session, injects context
- `ConversationManager.chat(user_input, history_override=None) -> str` — single turn
- `ConversationManager.stream_chat(user_input) -> Generator[str]` — streaming variant

### `modules/face_tracking.py`
- `FaceTracker(reachy=None)` — `reachy=None` → simulation mode
- `FaceTracker.detect_face(frame) -> FacePosition | None`
- `FaceTracker.update(face)` — moves neck via proportional control
- Body rotation triggered when `|dx_norm| > 0.40`

### `modules/face_recognition_module.py`
- `FaceRecognitionModule.identify(frame) -> str | None` — returns name or `None`
- `FaceRecognitionModule.register_person(name, frame) -> bool` — adds encoding
- `FaceRecognitionModule.register_from_camera(name, camera_index)` — interactive

## CLI flags

```
python main.py                         # Full run (requires REACHY_IP)
python main.py --no-robot              # Simulated robot
python main.py --setup                 # Init DB only
python main.py --add-person "Alice"    # Register Alice's face
python main.py --camera 1              # Use camera device 1
```

## Common development tasks

### Add a new person without restarting
```bash
python main.py --add-person "Bob"
```

### Inspect the database
```bash
sqlite3 reachy_memory.db ".tables"
sqlite3 reachy_memory.db "SELECT * FROM persons;"
sqlite3 reachy_memory.db "SELECT * FROM conversations ORDER BY id DESC LIMIT 20;"
```

### Test a module in isolation
```bash
python -m modules.conversation           # interactive chat test
python -m modules.face_tracking --no-robot   # webcam tracking visualiser
python -m modules.face_recognition_module --identify  # live recognition
```

### Reset all memory
```bash
rm reachy_memory.db face_encodings.pkl
python main.py --setup
```

## GPT summarization

After **50 messages** per person, `memory.py` automatically calls GPT-4o to
produce a ≤200-word summary stored in the `summaries` table.  The latest
summary is injected into every subsequent system prompt so the robot retains
long-term context without exceeding the context window.

## Face recognition confidence

`FACE_CONFIDENCE_THRESHOLD` is the **maximum face distance** accepted as a
match (lower = stricter).  Default `0.5` works well in normal lighting.
Reduce to `0.4` to avoid false positives; increase to `0.6` if known people
are being rejected.

## Robot SDK notes

Uses `reachy2-sdk`.  Key objects accessed:
- `reachy.head.neck.yaw.goal_position` — horizontal neck rotation (°)
- `reachy.head.neck.pitch.goal_position` — vertical neck tilt (°)
- `reachy.mobile_base.set_speed(vx, vy, vtheta)` — body rotation

## Dependencies

Install with:
```bash
pip install -r requirements.txt
```

On macOS, install `cmake` first (required by dlib, a dependency of `face_recognition`):
```bash
brew install cmake
```
