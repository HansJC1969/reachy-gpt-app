# Reachy GPT App

Connect a **Reachy Mini** robot to **OpenAI GPT-4o** for natural conversation with:

- Real-time face tracking (head + body follow)
- Face recognition — remembers people and animals by name
- Persistent memory via SQLite (last 10 messages + GPT-generated summaries)

---

## Prerequisites

### macOS
```bash
# dlib (required by face_recognition) needs cmake
brew install cmake
```

### All platforms
- Python 3.10+
- A webcam
- OpenAI API key

---

## Installation

```bash
# 1. Clone
git clone https://github.com/your-org/reachy-gpt-app.git
cd reachy-gpt-app

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.template .env
# Edit .env with your OPENAI_API_KEY and (optionally) REACHY_IP
```

---

## First run

### 1. Initialise the database
```bash
python main.py --setup
```

### 2. Register yourself (optional but recommended)
```bash
python main.py --add-person "Your Name"
# A camera window will open — look at it while 5 samples are captured.
```

### 3. Start the app (without a physical robot)
```bash
python main.py --no-robot
```

### 4. Start with a real Reachy
```bash
# Make sure REACHY_IP is set in .env
python main.py
```

---

## CLI reference

| Flag | Description |
|---|---|
| `--no-robot` | Run without a physical Reachy (simulation mode) |
| `--setup` | Initialise the SQLite database and exit |
| `--add-person "Name"` | Register a new person's face via camera |
| `--camera INDEX` | Camera device index (default: `CAMERA_INDEX` env var or `0`) |

---

## Usage

Once the app is running, type into the terminal to talk to Reachy:

```
[Stranger] You: Hello!
Reachy: Hi there! Great to meet you. What's on your mind today?

[Alice] You: Do you remember what we talked about last time?
Reachy: Of course! Last time we chatted about your trip to Japan and your love of robotics.
```

The robot will automatically:
- Track your face with its neck and body
- Recognise you after the first few frames
- Load your conversation history from the database
- Generate a summary every 50 messages to maintain long-term context

---

## Testing modules independently

```bash
# GPT chat only (no camera, no robot)
python -m modules.conversation

# Face tracking visualiser (webcam required)
python -m modules.face_tracking --no-robot

# Live face recognition
python -m modules.face_recognition_module --identify

# List known people in the encodings file
python -m modules.face_recognition_module
```

---

## Project structure

```
reachy-gpt-app/
├── main.py                          # Entry point + thread orchestration
├── modules/
│   ├── conversation.py              # OpenAI GPT-4o chat manager
│   ├── face_tracking.py             # Haar cascade detection + neck/body control
│   ├── face_recognition_module.py   # Face encoding + identification
│   └── memory.py                    # SQLite persistence + auto-summarization
├── requirements.txt
├── .env.template
├── CLAUDE.md                        # Developer reference for Claude Code
└── README.md
```

---

## Environment variables

Copy `.env.template` to `.env` and fill in:

```env
OPENAI_API_KEY=sk-...           # Required
REACHY_IP=192.168.1.100         # Required for robot mode
CAMERA_INDEX=0                  # Optional, default 0
FACE_CONFIDENCE_THRESHOLD=0.5   # Optional, default 0.5
```

---

## Troubleshooting

**`dlib` fails to install on macOS**
```bash
brew install cmake
pip install dlib
pip install face_recognition
```

**Camera not found**
```bash
# Try a different index
python main.py --no-robot --camera 1
```

**Robot not connecting**
```bash
# Verify IP and that Reachy is powered on and on the same network
ping 192.168.1.100
```

**Reset all memory**
```bash
rm reachy_memory.db face_encodings.pkl
python main.py --setup
```

---

## License

MIT
