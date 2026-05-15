# Reachy GPT App — Claude Code Reference

## Project overview

Connect a **Reachy Mini** robot (Pollen Robotics) to **OpenAI GPT-4o** for
natural conversation, with:

- Real-time **face tracking** — head via `look_at_image()` + body yaw rotation
- **Face recognition** — remembers specific people and animals by name
- **Persistent memory** via SQLite — last 10 messages + GPT-generated summary per person

## Repository layout

```
reachy-gpt-app/
├── main.py                          # Entry point; orchestrates all threads
├── modules/
│   ├── __init__.py
│   ├── conversation.py              # GPT-4o chat + tool-execution loop
│   ├── emotions.py                  # Keyframe animations: neck + antennas; move_head()
│   ├── face_tracking.py             # OpenCV Haar cascade + neck control
│   ├── face_recognition_module.py   # face_recognition lib + pickle storage
│   ├── memory.py                    # SQLite: persons / conversations / summaries
│   ├── profiles.py                  # Personality profile loader (instructions + voice)
│   ├── speech.py                    # OpenAI TTS: queue-based, SDK + sounddevice backends
│   ├── stt.py                       # OpenAI Whisper STT + DoA/RMS VAD
│   ├── vision.py                    # GPT-4o vision: scene description
│   └── websearch.py                 # Web search: Tavily (primary) / DuckDuckGo
├── profiles/
│   ├── default/                     # Standard Reachy — warm, curious, coral voice
│   ├── captain_circuit/             # Pirate robot — echo voice
│   ├── hype_bot/                    # Enthusiastic hype machine — shimmer voice
│   ├── noir_detective/              # Hard-boiled noir detective — onyx voice
│   ├── mars_rover/                  # Space-exploration scientist — alloy voice
│   ├── mad_scientist/               # Eccentric Professor Zap — fable voice
│   └── time_traveler/               # Temporal explorer — sage voice
├── requirements.txt
├── .env.template                    # Copy → .env and fill values
├── CLAUDE.md                        # This file
└── README.md
```

## Environment variables (`.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | — | OpenAI API key |
| `CAMERA_INDEX` | no | `0` | OpenCV camera device index |
| `FACE_CONFIDENCE_THRESHOLD` | no | `0.5` | Lower = stricter face recognition |
| `ENCODINGS_PATH` | no | `face_encodings.pkl` | Path to pickle file |
| `DB_PATH` | no | `reachy_memory.db` | SQLite database path |

> `REACHY_IP` is no longer used. The `reachy-mini` SDK auto-detects USB (Lite) and WiFi (Wireless) connections.

## Thread architecture (`main.py`)

```
camera_loop        (daemon)  → captures frames at ~30 fps
tracking_loop      (daemon)  → moves neck at 20 Hz using latest face position
recognition_loop   (daemon)  → identifies person every 15 frames
vision_loop        (daemon)  → GPT-4o scene description every VISION_INTERVAL s
idle_loop          (daemon)  → plays MÜDE after 12 s without a detected face
conversation_loop  (main)    → stdin → GPT-4o (tools) → stdout + emotion
```

All threads share `SharedState` which uses `threading.Lock` for safe access.
`stop_event` (a `threading.Event`) signals all threads to exit cleanly.

## Emotion system (`modules/emotions.py`)

### Supported emotions

| Enum value | German | Bewegung |
|---|---|---|
| `NEUTRAL` | Neutral | Kopf zentriert, Antennen waagerecht |
| `FREUDE` | Freude | Schnelle Kopfbewegungen auf/ab, Antennen federn hoch und wackeln |
| `TRAUER` | Trauer | Kopf sinkt langsam vorwärts, Antennen hängen nach unten |
| `ANGST` | Angst | Kopf zittert links/rechts schnell, Antennen gedrückt nach unten |
| `MÜDE` | Müde | Sehr langsames Abnicken, kurzes Aufschrecken mid-Animation |
| `NACHDENKEN` | Nachdenken | Kopf kippt rechts + oben, linke Antenne hoch, kleine Schwingungen |
| `TANZEN` | Tanzen | Rhythmischer Links-Rechts-Schwung (4×), Antennen gegenläufig |
| `ÜBERRASCHUNG` | Überraschung | Schneller Ruck nach hinten, beide Antennen schnellen hoch |
| `NEUGIER` | Neugier | Kopf neigt sich vor und zur Seite, Antennen gleichmäßig hochgerichtet |

### Hardware interface (reachy-mini SDK)
`_apply()` in `EmotionEngine` calls:
```python
from reachy_mini.utils import create_head_pose
head_pose = create_head_pose(pitch=°, yaw=°, roll=°, degrees=True)   # 4×4 matrix
antennas  = np.deg2rad([r_ant_deg, l_ant_deg])   # right first, radians
reachy.set_target(head=head_pose, antennas=antennas)
```
Keyframe values: pitch (°), yaw (°), roll (°), l_ant (°), r_ant (°) — all degrees.

### Key classes

```python
EmotionEngine(reachy=None)          # None → simulation mode
engine.play(Emotion.FREUDE)         # non-blocking
engine.play(Emotion.TANZEN, block=True)  # blocking
engine.stop()                       # sofort stoppen + neutral
engine.idle()                       # sanft zu neutral

parse_emotion("freude") → Emotion.FREUDE   # string → Enum
```

### Keyframe format

```python
Keyframe(t=0.5, pitch=10, yaw=5, roll=3, l_ant=45, r_ant=40)
# t     = Zeit in Sekunden ab Animationsstart
# pitch = Neigung °  (±20 typisch)
# yaw   = Drehung °  (±30 typisch)
# roll  = Kippen °   (±15 typisch)
# l_ant = linke  Antenne ° (-45 unten … +65 hoch)
# r_ant = rechte Antenne ° (-45 unten … +65 hoch)
```

Interpolation erfolgt mit Cosinus-Glättung zwischen Keyframes.

### Emotion detection (GPT function calling)

`ConversationManager.chat_with_emotion()` nutzt einen OpenAI Function-Call
`express_emotion(emotion: str)` im selben API-Request — keine zweite
Anfrage nötig. Falls GPT keine Emotion mitschickt, wird `NEUTRAL` zurückgegeben.

### CLI flag
```bash
python main.py --emotion-test      # alle Animationen nacheinander abspielen
python -m modules.emotions --demo  # dasselbe ohne Roboter
python -m modules.emotions --demo --emotion tanzen  # einzelne Emotion
```

## Web search (`modules/websearch.py`)

```python
searcher = WebSearcher()           # auto-detects Tavily or DuckDuckGo
results  = searcher.search("query", max_results=5)   # → list[SearchResult]
text     = searcher.format_results(results)           # numbered markdown
text     = searcher.search_and_format("query")        # convenience one-liner
searcher.backend                   # "tavily" | "duckduckgo" | "none"
```

Backend selection (in order of preference):
1. **Tavily** — set `TAVILY_API_KEY` in `.env` and `pip install tavily-python`
2. **DuckDuckGo** — `pip install duckduckgo-search` (no key required)

GPT decides autonomously when to call web search based on the question.
Triggers: current events, weather, prices, facts that may have changed.

```bash
python -m modules.websearch "Was ist die Hauptstadt von Japan?"
```

## Vision / scene recognition (`modules/vision.py`)

```python
vision = VisionAnalyzer(interval=10.0)
desc   = vision.analyze(frame)                       # one-shot description
desc   = vision.analyze_periodic(frame)              # only if interval elapsed
desc   = vision.analyze_on_command(frame, question)  # answer specific question
vision.last_description                              # cached last result
vision.reset_timer()                                 # force next periodic run
```

Two integration points:
1. **Passive** — `vision_loop` thread calls `analyze_periodic()` every 10 s.
   The description is injected into the system prompt so Reachy always has
   background scene awareness.
2. **Active** — GPT calls the `get_visual_description` tool when the user asks
   "was siehst du?" or when visual context would improve the answer.

```bash
python -m modules.vision --camera 0 --interval 10
```

## GPT tool execution flow (`modules/conversation.py`)

GPT-4o has access to three tools per reply:

| Tool | Trigger | Execution |
|---|---|---|
| `express_emotion` | every reply | captures emotion, returns "ok" |
| `web_search` | current facts needed | calls `WebSearcher.search_and_format()` |
| `get_visual_description` | visual question | calls `VisionAnalyzer.analyze_on_command()` |

The `chat_with_emotion()` method runs a tool loop (max 5 rounds):
```
GPT response
  └─ tool_calls?
      ├─ execute search / vision → append result → next GPT call
      ├─ execute express_emotion → capture emotion → continue
      └─ finish_reason == "stop" → return (reply, emotion)
```

`convo.set_latest_frame(frame)` must be called before `chat_with_emotion()`
so the vision tool has a fresh frame available.

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

## CLI flags

```
python main.py                              # Full run
python main.py --no-robot                  # Simulated robot (no Reachy hardware)
python main.py --profile captain_circuit   # Use a personality profile
python main.py --voice onyx                # Override TTS voice
python main.py --no-vision                 # Disable GPT-4o scene analysis
python main.py --no-search                 # Disable web search
python main.py --setup                     # Init DB only
python main.py --add-person "Alice"        # Register Alice's face
python main.py --camera 1                  # Use camera device 1
python main.py --emotion-test              # Demo all emotion animations
python main.py --text-input                # Keyboard input instead of microphone
```

## Personality profiles

Profiles live in `profiles/<name>/` with two files:
- `instructions.txt` — personality text injected into every system prompt
- `voice.txt` — optional TTS voice override

| Profile | Character | Voice |
|---|---|---|
| `default` | Warm, curious, lightly witty Reachy | coral |
| `captain_circuit` | Pirate robot — "aye matey!" | echo |
| `hype_bot` | Ultra-enthusiastic motivator | shimmer |
| `noir_detective` | Hard-boiled noir detective | onyx |
| `mars_rover` | Space-exploration scientist | alloy |
| `mad_scientist` | Eccentric Professor Zap | fable |
| `time_traveler` | Temporal explorer from many eras | sage |

### Adding a custom profile
```bash
mkdir profiles/my_robot
echo "Personality: ..." > profiles/my_robot/instructions.txt
echo "nova" > profiles/my_robot/voice.txt
python main.py --profile my_robot
```

## GPT tools available to the model

| Tool | Trigger | Execution |
|---|---|---|
| `express_emotion` | every reply | captures emotion → EmotionEngine.play() |
| `web_search` | current facts needed | Tavily / DuckDuckGo |
| `move_head` | show attention or reaction | EmotionEngine.move_head(direction) |
| `dance` | celebration / music / fun | EmotionEngine.play(TANZEN) |
| `get_visual_description` | visual question | VisionAnalyzer.analyze_on_command() |

## Robot SDK notes

Uses `reachy-mini` (`from reachy_mini import ReachyMini`).

**Connection** — SDK auto-detects USB (Lite) and WiFi (Wireless):
```python
from reachy_mini import ReachyMini
# media_backend="no_media" releases camera/audio so OpenCV/sounddevice can use them
reachy = ReachyMini(media_backend="no_media")
reachy.__enter__()   # connect
# … run …
reachy.__exit__(None, None, None)  # disconnect cleanly
```

**Head tracking** — pixel-based; SDK handles inverse kinematics:
```python
reachy.look_at_image(u, v, duration=0)   # snap head to face pixel (cx, cy)
```

**Head + antennas** (keyframe animations):
```python
from reachy_mini.utils import create_head_pose
import numpy as np

head_pose = create_head_pose(pitch=10, yaw=5, roll=3, degrees=True)  # 4×4 matrix
antennas  = np.deg2rad([right_deg, left_deg])   # SDK order: right first
reachy.set_target(head=head_pose, antennas=antennas)   # instant
reachy.goto_target(head=head_pose, antennas=antennas, duration=0.5)  # interpolated
```

**Body yaw** — absolute angle in radians:
```python
reachy.set_target_body_yaw(rad)        # instant
reachy.goto_target(body_yaw=rad, duration=0.5)  # smooth
```

**Safety limits** (SDK clamps automatically):
| Joint | Range |
|---|---|
| Head pitch / roll | ±40° |
| Head yaw | ±180° |
| Body yaw | ±160° |

## Dependencies

Install with:
```bash
pip install -r requirements.txt
```

On macOS, install `cmake` first (required by dlib, a dependency of `face_recognition`):
```bash
brew install cmake
```
