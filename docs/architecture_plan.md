# Joi: Spatial AI Companion Architecture Blueprint (v1.3 - FAANG Grade & Dockerized)

> **A real-time, low-latency, multimodal spatial AI companion inspired by Joi from *Blade Runner 2049*.**  
> Architected as a **production-grade, 90% Python Clean Architecture / Ports-and-Adapters system**, containerized via **Docker**, featuring CPU-offloaded neural speech synthesis and transcription, BYOK (Bring-Your-Own-Key) LLM streaming, SQLite FTS5 episodic memory, structured observability, and a thin Three.js VRM display actuator supporting transparent desktop holograms and Android WebXR spatial grounding.

---

## 1. System Vision & Behavioral Contract

### 1.1 The "Joi" Experience
Joi is designed as an **embodied spatial presence** rather than a passive chatbot:
- **Physicality & Posture Awareness:** Adapts posture to detected surfaces (sitting on desk edges, couches, or standing) and reacts with natural idle behaviors (leaning, glancing away, shifting weight).
- **Sub-Second Voice Cadence:** Conversational latency targeted at **~450ms (Text-to-Voice)** and **~1.0s – 1.4s (Voice-to-Voice)** using streaming sentence-chunked neural TTS and instant visual emotion cues.
- **Environmental Context:** Aware of the user's workspace, time of day, and physical surroundings.
- **Provider-Agnostic Freedom:** Complete BYOK (Bring-Your-Own-Key) architecture supporting OpenRouter, Venice.ai, RunPod, or local quantized models (Ollama / llama.cpp) without vendor lock-in.

### 1.2 Tag Vocabulary & Parser Schema
*(Note: Full personality prompt tuning is deferred until the voice loop is live. The tag vocabulary below defines the contractual interface between the LLM and the 3D rig).*

- **Expression Tags:** `[smile]`, `[curious]`, `[thoughtful]`, `[smirk]`, `[concerned]`, `[giggle]`, `[neutral]`
- **Action & Posture Tags:** `[sit]`, `[stand]`, `[lean_forward]`, `[look_away]`, `[wave]`, `[nod]`
- **Inline Streaming Syntax:** `"[smile] [lean_forward] Hey Ken. Still wrestling with that compiler bug?"`

---

## 2. FAANG-Style Clean Architecture (90% Python / 10% Thin Actuator)

To achieve high testability, maintainability, and zero vendor lock-in, the system follows **Hexagonal / Clean Architecture (Ports and Adapters)** with strict **Dependency Injection** and **Pydantic V2 Type Safety**.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        HEADLESS CORE ENGINE (CONTAINERIZED / DOCKER)                   │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              API / INGRESS LAYER                                 │  │
│  │  - FastAPI Routes (/healthz, /metrics)   - Dependency Injection Container        │  │
│  │  - Full-Duplex WebSocket Handler (/ws)    - Static Frontend Web Server           │  │
│  └────────────────────────────────────────┬─────────────────────────────────────────┘  │
│                                           │ Calls                                      │
│  ┌────────────────────────────────────────▼─────────────────────────────────────────┐  │
│  │                           APPLICATION / SERVICES LAYER                           │  │
│  │  - TurnOrchestrator: Coordinates LLM, STT, TTS, Visemes, and Barge-In            │  │
│  │  - StreamingParser: Intercepts inline [tags] & buffers sentence boundaries       │  │
│  │  - MemoryService: SQLite FTS5 search & async fact extraction                     │  │
│  └────────────────────┬───────────────────────────────────────┬─────────────────────┘  │
│                       │ Defines Interfaces                    │ Implements             │
│  ┌────────────────────▼────────────────────┐  ┌───────────────▼─────────────────────┐  │
│  │              DOMAIN LAYER               │  │        INFRASTRUCTURE / ADAPTERS    │  │
│  │  - Protocols (TTSEngine, STTEngine,     │  │  - KokoroTTSAdapter (ONNX on CPU)   │  │
│  │    LLMGateway, MemoryRepository)        │  │  - WhisperSTTAdapter (INT8 on CPU)  │  │
│  │  - Domain Models (TurnState, Fact)      │  │  - OpenRouterAdapter (SSE Streamer) │  │
│  │  - Strict Wire DTOs (Events & Packets)  │  │  - SQLiteFTSAdapter (BM25 Search)   │  │
│  └─────────────────────────────────────────┘  └─────────────────────────────────────┘  │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                        CORE FOUNDATION (Cross-Cutting)                           │  │
│  │  - Pydantic BaseSettings (12-Factor App)   - Structured JSON Logger (Trace IDs)  │  │
│  │  - Lifespan & Cancellation Token Pools     - Domain Exception Hierarchy          │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ WebSocket (Dual-Channel over LAN/HTTPS)
                        ┌───────────────────┴───────────────────┐
                        ▼                                       ▼
           Client A: PC Desktop Overlay             Client B: Android Mobile
           - PySide6 Frameless Shell / WebGL        - Android Chrome WebXR
           - Three.js (@pixiv/three-vrm)            - Surface plane hit-testing
           - Web Audio API (Buffered Queue)         - WebRTC AEC microphone capture
           - Browser MediaPipe Tasks (60 FPS)       - Retargeted animation cross-fade
           - MToon Fresnel & scanline shaders       - Furniture depth occlusion
```

---

## 3. Containerization & Deployment Architecture (Docker)

The **entire headless Python backend is packaged into a self-contained Docker container**, decoupling heavy inference and memory storage from local workstation operating systems.

### 3.1 Why CPU-Bound Speech Makes Docker Seamless
Because Kokoro-82M TTS and Faster-Whisper run on **CPU via ONNX/INT8**, the container:
- **Requires zero NVIDIA Container Toolkit or GPU pass-through drivers** to run speech.
- Runs identically on Linux, Windows (WSL2 / Docker Desktop), or remote VPS/home servers.
- Leaves host GPU VRAM 100% untouched for local 3D rendering or local LLM inference.

### 3.2 Persistent Volume Layout
The container mounts two persistent host directories to prevent re-downloading model weights or losing conversation history across image rebuilds:
1. `/app/models`: Cached ONNX model weights (`kokoro-v1.0.onnx`, `voices-v1.0.bin`, Faster-Whisper INT8 cache).
2. `/app/data`: SQLite database (`joi_memory.db`) for persistent episodic FTS5 memory.

### 3.3 Production `docker-compose.yml` Specification
```yaml
services:
  joi-core:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: joi-core-engine
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_BASE=${OPENAI_API_BASE:-https://openrouter.ai/api/v1}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MODEL_NAME=${MODEL_NAME:-cognitivecomputations/dolphin-mixtral-8x7b}
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"]
      interval: 15s
      timeout: 5s
      retries: 3
```

---

## 4. Real-Time Audio Pipeline & Latency Contract

### 4.1 Latency Budget

| Interaction Mode | Pipeline Breakdown | Target Latency |
| :--- | :--- | :--- |
| **Text-in $\rightarrow$ First Audio** | LLM TTFT (300ms) + Kokoro Sentence 1 (150ms) | **~450ms** |
| **Voice-in $\rightarrow$ First Audio** | VAD Endpointing (350ms) + Faster-Whisper (200ms) + LLM TTFT (300ms) + Kokoro (150ms) | **~1.0s – 1.4s** |

### 4.2 Full-Duplex Audio & Ingress
- **Client-Side VAD & AEC:** Microphone audio is captured with `echoCancellation: true` to prevent Joi from hearing her own voice. Client-side VAD detects speech initiation and sends an immediate `{"type": "barge_in"}` event.
- **Audio Uplink:** On speech completion, the client transmits raw PCM bytes:
  `[uint32 message_type = 0x01][uint32 sample_rate = 16000][16-bit PCM bytes]`
  The Python server transcribes it via Faster-Whisper (INT8 on CPU) and feeds the text into the turn orchestrator.
- **8-Byte Binary Frame Header:**
  `[uint32 turn_id (4 bytes)][uint32 seq_num (4 bytes)][Raw 16-bit PCM Audio Payload (24kHz Mono)]`
  If a turn is cancelled, in-flight chunks are dropped immediately by the client.
- **Threadpool Cancellation:** Kokoro ONNX synthesis runs with an `asyncio.Event` cancellation check between sentences, terminating generation within ~100ms of interruption.

---

## 5. Hologram Aesthetics & 3D Rigging

### 5.1 MToon Shader Chunk Injection
Rather than using screen-space post-processing (`EffectComposer`), which breaks inside WebXR framebuffers, we inject custom GLSL directly into `MToonMaterial.onBeforeCompile`:
1. **Fresnel Rim Glow:** Amplifies light at glancing angles to produce a volumetric hologram look while preserving facial blendshapes and morph targets.
2. **World-Space Scanlines:** Subtle horizontal raster lines mapped to world Y coordinates.

### 5.2 Authored Animation Blending
Instead of unstable runtime Inverse Kinematics (IK), we retarget clean humanoid animation clips (.vrma / Mixamo) and cross-fade them via Three.js `AnimationMixer` over 400ms when posture tags (`[sit]`, `[stand]`) arrive.

### 5.3 60 FPS Head-Tracking Parallax (Client-Side)
PC Desktop overlay executes **MediaPipe Tasks Vision for Web** (`@mediapipe/tasks-vision`) directly in JavaScript at **30–60 FPS**, applying exponential moving average (EMA) lerping to the Three.js camera. Python handles zero webcam frames, eliminating network lag.

---

## 6. Episodic Memory Core: SQLite FTS5

- **Storage:** Local SQLite database utilizing the `fts5` full-text search module, mounted to `/app/data/joi_memory.db`.
- **Retrieval:** BM25 lexical ranking with recency scoring against the current user message.
- **Async Extraction:** A background task extracts persistent user facts (projects, preferences, habits) at the end of each turn, validates deduplication, and stores them asynchronously.

---

## 7. WebSocket Wire Protocol Specification

### 7.1 Server $\rightarrow$ Client (JSON Control Plane)
```json
// Turn Initiation
{ "type": "turn_start", "turn_id": 1, "trace_id": "c9bf9e57-1685-4c89-bafb-ff5af830be8a" }

// Instant Facial Expression
{ "type": "expression", "turn_id": 1, "data": { "blendshape": "smile", "intensity": 0.8, "duration_ms": 1200 } }

// Posture / Action Trigger
{ "type": "action", "turn_id": 1, "data": { "animation": "Sitting_Desk", "fade_ms": 400 } }

// Turn Interruption & Completion
{ "type": "turn_cancelled", "turn_id": 1 }
{ "type": "turn_end", "turn_id": 1 }
```

### 7.2 Server $\rightarrow$ Client (Binary Audio Data Plane)
Raw binary frame sent via `websocket.send_bytes(...)`:
- `[Bytes 0-3]`: `turn_id` (uint32, little-endian)
- `[Bytes 4-7]`: `seq_num` (uint32, little-endian)
- `[Bytes 8+]`: Linear PCM 16-bit audio, 24,000 Hz, Mono

### 7.3 Client $\rightarrow$ Server (JSON Control Plane)
```json
// Text Message (Typing / Terminal Client)
{ "type": "user_message", "text": "Hey Joi, how's the build going?" }

// Immediate Barge-In (User spoke mid-sentence)
{ "type": "barge_in" }

// Spatial Telemetry (Android WebXR plane info)
{ "type": "spatial_telemetry", "data": { "plane_height": 0.45, "ambient_light": 0.7 } }
```

### 7.4 Client $\rightarrow$ Server (Binary Voice Uplink)
Raw binary frame sent via `websocket.send_bytes(...)`:
- `[Bytes 0-3]`: Message type `0x01` (User speech audio)
- `[Bytes 4-7]`: Sample rate (e.g. 16,000 Hz)
- `[Bytes 8+]`: Linear PCM 16-bit audio from microphone

---

## 8. FAANG-Style Repository Layout

```
joi/
├── Dockerfile                     # Multi-stage production container build (uv-based)
├── docker-compose.yml             # Single-command orchestration with volume mounts
├── .dockerignore                  # Lean build context exclusions
├── pyproject.toml                 # uv package management, ruff, mypy, pytest configs
├── .env.example                   # Strongly typed environment variable template
├── docs/                          # Architecture blueprints & technical documentation
│   └── architecture_plan.md       # Master specification (v1.3)
│
├── models/                        # Host volume mount: Cached ONNX & speech models
├── data/                          # Host volume mount: SQLite FTS5 database
│
├── src/joi/                       # Production Python package (90% of codebase)
│   ├── __init__.py
│   ├── main.py                    # Application factory (create_app) & lifespan hooks
│   │
│   ├── core/                      # Foundational cross-cutting utilities
│   │   ├── config.py              # Pydantic BaseSettings (12-Factor App)
│   │   ├── logging.py             # Structured JSON logger with trace/turn context
│   │   └── exceptions.py          # Domain-specific exception hierarchy
│   │
│   ├── domain/                    # Pure business logic & contracts (Zero framework code)
│   │   ├── models.py              # Core entities (TurnState, MemoryFact, EmotionTag)
│   │   ├── protocols.py           # Abstract Protocols: LLMGateway, TTSEngine, STTEngine
│   │   └── events.py              # Pydantic wire models (ClientEvent, ServerEvent)
│   │
│   ├── services/                  # Application orchestrators & state machines
│   │   ├── orchestrator.py        # Central turn coordinator (LLM + TTS + Visemes + Barge-in)
│   │   ├── streaming_parser.py    # Tag-extraction & punctuation state machine
│   │   └── memory_service.py      # SQLite FTS5 episodic storage & fact extraction
│   │
│   ├── infrastructure/            # Concrete hardware & external adapters
│   │   ├── llm/
│   │   │   └── openrouter_adapter.py   # OpenAI/OpenRouter SSE streaming client
│   │   ├── speech/
│   │   │   ├── kokoro_tts_adapter.py   # Kokoro ONNX model downloader & runner
│   │   │   └── whisper_stt_adapter.py  # Faster-Whisper INT8 model runner
│   │   └── persistence/
│   │       └── sqlite_fts_repository.py # SQLite FTS5 database driver
│   │
│   └── api/                       # Ingress & presentation layer
│       ├── dependencies.py        # FastAPI Dependency Injection wiring
│       ├── routes/
│       │   ├── health.py          # /healthz and /metrics endpoints
│       │   └── websocket.py       # Full-duplex /ws connection handler & router
│       └── desktop/
│           └── overlay_app.py     # PySide6 transparent window desktop launcher
│
├── frontend/                      # Thin 3D Display Actuator (< 250 lines JS)
│   ├── index.html                 # HTML5 canvas shell & WebXR bootstrap
│   ├── style.css                  # Minimal transparent styling
│   ├── js/
│   │   ├── app.js                 # WS protocol & binary header parser
│   │   ├── audio_controller.js    # Web Audio scheduler, AEC mic & client VAD
│   │   ├── avatar.js              # Three.js VRM loader & AnimationMixer
│   │   ├── shaders.js             # MToon Fresnel & scanline GLSL
│   │   └── head_tracker.js        # MediaPipe Tasks 60 FPS camera lerp
│   └── assets/                    # .vrm avatars and .vrma animation clips
│
├── tests/                         # Pytest test suite
│   ├── conftest.py                # Test fixtures & mock LLM/TTS/STT providers
│   ├── unit/                      # Unit tests (tag parser, state machine, audio header)
│   │   ├── test_streaming_parser.py
│   │   └── test_binary_protocol.py
│   └── integration/               # Integration tests (full WebSocket turn loop)
│       └── test_websocket_turn.py
│
└── scripts/                       # Developer toolchain
    ├── download_models.py         # Standalone model asset pre-fetcher
    └── mock_client.py             # Headless Python terminal client for latency benchmarking
```

---

## 9. Phased Implementation Roadmap

- [ ] **Phase 1: Domain Models, Protocols & Speech Infrastructure**
  - Scaffold directory structure, `Dockerfile`, `docker-compose.yml`, and `pyproject.toml`.
  - Implement `core/config.py`, `core/logging.py`, and `domain/protocols.py`.
  - Implement `infrastructure/speech/kokoro_tts_adapter.py` (auto-download weights, warmup, cancellation token).
  - Implement `infrastructure/speech/whisper_stt_adapter.py` (Faster-Whisper INT8 on CPU).
  - Implement `services/streaming_parser.py` (tag-extraction state machine & sentence boundary buffer).
  - Verify unit tests with `pytest`.

- [ ] **Phase 2: Turn Orchestration & WebSocket Hub**
  - Implement `infrastructure/llm/openrouter_adapter.py` (BYOK OpenAI SSE client).
  - Implement `services/orchestrator.py` (full-duplex lifecycle, barge-in, turn state).
  - Implement `api/routes/websocket.py` and `api/dependencies.py`.
  - Validate the complete voice loop using `scripts/mock_client.py`.

- [ ] **Phase 3: Thin 3D Canvas, Shaders & Audio Visemes**
  - Scaffold `frontend/` with Three.js and `@pixiv/three-vrm`.
  - Implement `audio_controller.js` for Web Audio scheduling and mic VAD uplink.
  - Inject Fresnel glow and scanline GLSL into `MToonMaterial` via `onBeforeCompile`.
  - Connect `AnimationMixer` to crossfade idle and seated animations on action tags.

- [ ] **Phase 4: Docker Containerization, Desktop Overlay & WebXR**
  - Verify `docker compose up --build` runs the complete headless hub cleanly.
  - PC: Launch frameless transparent window with PySide6 connecting to the container.
  - PC: Enable browser-based MediaPipe head tracking for 60 FPS parallax.
  - Mobile: Enable WebXR hit-testing over Tailscale HTTPS for Android Chrome.
  - Implement `services/memory_service.py` with SQLite FTS5 BM25 search.
