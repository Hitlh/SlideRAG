<div align="center">
<h2 align="center">
  <img src="./assets/mylogo.png" width="180" alt="SlideRAG Logo">
</h2>
<h2 align="center">
  <b>SlideRAG: PPT-Centric Multimodal RAG for Study Preview & Exam Review</b>
</h2>
<div>
<a href="./README.md"><b>English</b></a> | <a href="./README_zh.md">简体中文</a>
</div>
<br/>
<div align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-2b6cb0" alt="Python">
    <img src="https://img.shields.io/badge/RAG-Multimodal-0f766e" alt="Multimodal RAG">
    <img src="https://img.shields.io/badge/Agent-Tool%20Calling-b45309" alt="Agent Tool Calling">
    <img src="https://img.shields.io/badge/Channels-Web%20%7C%20QQ%20%7C%20Feishu%20%7C%20WeChat%20%7C%20WhatsApp-334155" alt="Channels">
</div>
</div>

## :new: Updates
- [04/2026] :fire: SlideRAG is now open-source.

## :rocket: SlideRAG
🎓 SlideRAG is an end-to-end assistant for understanding PPT/PPTX files as multimodal learning materials.

🧠 Unlike text-only QA systems, SlideRAG treats each slide as a structured multimodal unit and combines parsing, retrieval, and agent tool-calling.

📌 It is designed for two key learning scenarios: before-class preview and before-exam review.

## 🎬 Demo Showcase

This video presents SlideRAG from a product perspective and walks through the full user flow: upload slides, inspect parsing results, and run multi-turn QA over PPT content.

https://github.com/user-attachments/assets/09f12095-fb4e-4cf7-8bc1-ea148d23add9

Here is an example that demonstrates the performance of SlideRAG in understanding the content and structure of multimodal slides:

<table>
<tr>
<td><img src="./assets/example_03.png" width="100%" alt="Demo 3"></td>
<td><img src="./assets/example_04.png" width="100%" alt="Demo 4"></td>
<td><img src="./assets/example_05.png" width="100%" alt="Demo 5"></td>
</tr>
<tr>
<td><img src="./assets/example_06.png" width="100%" alt="Demo 6"></td>
<td><img src="./assets/example_08.png" width="100%" alt="Demo 8"></td>
<td><img src="./assets/example_09.png" width="100%" alt="Demo 9"></td>
</tr>
</table>

QA case snapshots:

<table>
<tr>
<td><img src="./assets/qa_case_01.png" width="100%" alt="QA Case 1"></td>
<td><img src="./assets/qa_case_02.png" width="100%" alt="QA Case 2"></td>
<td><img src="./assets/qa_case_03.png" width="100%" alt="QA Case 3"></td>
<td><img src="./assets/qa_case_04.png" width="100%" alt="QA Case 4"></td>
</tr>
</table>

## :sparkles: Key Features
- 🖼️ **PPT-first multimodal RAG pipeline**: Uses a unified multimodal parser and a graph-and-vector hybrid retrieval engine to support grounded QA across text, images, tables, and equations.
- 🪄 **Hidden-information expansion for concise slides**: Detects high-compression pages and expands implicit content into grounded explanatory text.
- 🔗 **Page-topic extraction and structural linking**: Extracts per-page topics and links related slides to model section-level continuity in long decks.
- 🤝 **Easy to use**: One backend supports Web, QQ, Feishu, WeChat, and WhatsApp bridge mode, making the assistant accessible in familiar study workflows.

## 🧩 Framework

<img src="./assets/index.png" >
<img src="./assets/agent_loop.png" >
SlideRAG follows a retrieval-augmented agent workflow:
1. Parse PPT/PPTX into typed multimodal items with page metadata.
2. Perform PPT-oriented enhancement (hidden-info expansion + topic extraction/linking).
3. Build unified multimodal knowledge storage for hybrid retrieval.
4. Use tool-calling agent loop to retrieve evidence and trigger optional image understanding.
5. Return grounded answers through Web/QQ/Feishu/WeChat/WhatsApp channels.

## 🚀 Quick Start

This section helps you run SlideRAG quickly for web usage, then optionally connect it to QQ, Feishu, WeChat, or WhatsApp.

### 1. Clone and install
```bash
git clone https://github.com/Hitlh/SlideRAG.git
cd SlideRAG

# Core dependencies
pip install -e .

# Optional channel dependencies
pip install -e .[qq]
pip install -e .[feishu]
pip install -e .[weixin]
pip install -e .[whatsapp]
pip install -e .[channels]
```

Because this project focuses on PPT understanding, install LibreOffice as an extra system dependency:
- Ubuntu/Debian: `sudo apt-get install libreoffice`
- Windows: download installer from the official website: https://www.libreoffice.org/
- macOS: `brew install --cask libreoffice`

### 2. Configure environment variables

Create a `.env` file by copying content from `env.project.example`, then fill in your keys and model settings.

> Note: If you want to tune advanced parser/context behavior (for example, `SUMMARY_LANGUAGE`, hidden-expansion options, and context window settings), edit the **Advanced parser/context options (optional)** section in `env.project.example`.

> Tip for users in mainland China: MinerU uses Hugging Face by default. If Hugging Face access is unstable, switch to ModelScope before running:
>
> ```bash
> export MINERU_MODEL_SOURCE=modelscope
> ```

#### 2.1 API keys and base models
```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=your_base_url

# Text and vision models used by SlideRAG pipeline
TEXT_LLM_MODEL=gpt-5.4
VLM_MODEL=gpt-5.4
```

#### 2.2 Agent model provider
```env
# Agent provider for rag_agent loop: openai | anthropic
AGENT_PROVIDER=openai
AGENT_MODEL=gpt-5.4

# Anthropic provider settings (only required when AGENT_PROVIDER=anthropic)
# If empty, runtime may fall back to OPENAI_API_KEY / OPENAI_BASE_URL.
ANTHROPIC_API_KEY=
ANTHROPIC_BASE_URL=
```

### 3. Start the web app
```bash
streamlit run client/app.py
```

After startup, open the Streamlit URL shown in terminal and start asking questions about your PPT files.

## Chat App Integration

SlideRAG can run the same QA agent through QQ, Feishu, WeChat, and WhatsApp bridge mode.

### QQ setup (requires QQ extras)

1. Create a QQ bot.
2. Go to the QQ Open Platform (https://q.qq.com/#/), sign in, and create your bot.
3. In your bot console, open "开发控制" and copy `APPID` and `APPSecret`.
4. Go to "沙箱配置", then in "在消息列表配置" choose "添加成员", enter your QQ number, and scan the QR code.
5. Put the files you want to chat with in a folder (default: `./uploaded_docs`).
6. Configure QQ environment variables:

```env
QQ_ENABLED=false                    # Set to true to enable QQ integration
QQ_APP_ID=                          # QQ bot APPID from the Open Platform
QQ_SECRET=                          # QQ bot APPSecret from the Open Platform
QQ_ALLOW_FROM=*                     # Allowed senders; use * to accept all
QQ_TARGET_FILE=                     # Default file to chat with; can switch via /file <filename>
QQ_UPLOADED_DOCS_DIR=./uploaded_docs # Directory that stores your source files

# Startup ready notification
QQ_STARTUP_NOTIFY_ENABLED=true      # Send a startup message when agent is ready
QQ_STARTUP_NOTIFY_MESSAGE=rag agent is ready. # Startup message content
QQ_STARTUP_NOTIFY_CHAT_ID=          # Target chat/user ID for startup notification
```

7. Run QQ backend:

```bash
python3 -m client.qq.runtime
# or
sliderag-qq
```

8. Start chatting. You can switch the active document with: `/file <filename>`.

### Feishu setup (requires Feishu extras)

1. Go to the Feishu/Lark open platform. China users can use open.feishu.cn, and global users can use open.larksuite.com.
2. Create an app and add the Bot capability.
3. In Developer Configuration > Permissions, enable the permissions needed for messaging:
  - im:message
  - im:message.p2p_msg:readonly
4. In Developer Configuration > Events and Callbacks, use long connection for event delivery, then add the event:
  - im.message.receive_v1
5. Save the App ID and App Secret in Credentials & Basic Information.
6. Publish the app.
7. Put the files you want to chat with in a folder (default: `./uploaded_docs`).
8. Configure Feishu environment variables:

```env
FEISHU_ENABLED=true
FEISHU_APP_ID=
FEISHU_APP_SECRET=
FEISHU_ENCRYPT_KEY=
FEISHU_VERIFICATION_TOKEN=
FEISHU_ALLOW_FROM=*
# feishu (China) | lark (global)
FEISHU_DOMAIN=feishu

# Optional startup target file for auto-ingest
FEISHU_TARGET_FILE=

# Runtime directories
FEISHU_UPLOADED_DOCS_DIR=./uploaded_docs
FEISHU_INGEST_OUTPUT_DIR=./output
FEISHU_RAG_WORKING_DIR=./rag_storage_by_feishu_file
FEISHU_RUNTIME_STATE_DIR=./rag_storage_feishu_runtime

# Startup ready notification
FEISHU_STARTUP_NOTIFY_ENABLED=true
FEISHU_STARTUP_NOTIFY_MESSAGE=agent is ready.
FEISHU_STARTUP_NOTIFY_CHAT_ID=
```

9. Run Feishu backend:

```bash
python3 -m client.feishu.runtime
```

10. Send one message in Feishu first if you want to use `FEISHU_ALLOW_FROM` or `FEISHU_STARTUP_NOTIFY_CHAT_ID`, then check runtime logs for the `sender_id` value.

### WeChat setup (requires WeChat extras)

1. Put the files you want to chat with in a folder (default: `./uploaded_docs`).
2. Configure WeChat environment variables:

```env
WEIXIN_ENABLED=true                # Set to true to enable WeChat integration
WEIXIN_ALLOW_FROM=*                # Allowed senders; use * to accept all
WEIXIN_TARGET_FILE=                # Default file to chat with; can switch via /file <filename>
WEIXIN_UPLOADED_DOCS_DIR=./uploaded_docs # Directory that stores your source files
WEIXIN_STARTUP_NOTIFY_ENABLED=true # Send a startup message when agent is ready
WEIXIN_STARTUP_NOTIFY_MESSAGE=agent is ready. # Startup message content
WEIXIN_STARTUP_NOTIFY_CHAT_ID=     # Target chat/user ID for startup notification
```

3. Run WeChat backend:

```bash
python3 -m client.weixin.runtime
# use -r to force re-login
python3 -m client.weixin.runtime -r
```

4. Scan QR code on first login.
5. Start chatting. You can switch the active document with: `/file <filename>`.

### WhatsApp setup (requires WhatsApp extras + local bridge)

SlideRAG uses a local WebSocket bridge with protocol messages (`message/status/qr/error`, `send/send_media`).

1. Start the bundled local WhatsApp bridge in this repository (default endpoint: `ws://127.0.0.1:3001`) and keep it running.
Example bridge startup:

```bash
cd /path/to/SlideRAG/bridge
npm install
npm run build

export BRIDGE_PORT=3001
export BRIDGE_TOKEN=replace_with_secret
export AUTH_DIR=/abs/path/to/wa_auth
npm start
```

`BRIDGE_TOKEN` must be the same value as `WHATSAPP_BRIDGE_TOKEN` in SlideRAG `.env`.
> Tip: If you test by sending messages from the same WhatsApp account (self-message),
> start bridge with `export WHATSAPP_ACCEPT_FROM_ME=true`.
2. Configure WhatsApp environment variables:

```env
WHATSAPP_ENABLED=true
WHATSAPP_ALLOW_FROM=*                      # Allowed sender/chat ids; use * to accept all
WHATSAPP_BRIDGE_URL=ws://127.0.0.1:3001    # Local bridge endpoint
WHATSAPP_BRIDGE_TOKEN=replace_with_secret  # Must match bridge auth token
WHATSAPP_RECONNECT_DELAY_S=5
WHATSAPP_SEND_RETRY_ATTEMPTS=3             # Outbound send retry attempts
WHATSAPP_SEND_RETRY_DELAY_MS=400           # Outbound retry delay in milliseconds
WHATSAPP_ACCEPT_GROUP_MESSAGES=true        # Accept group inbound messages
WHATSAPP_REQUIRE_MENTION_IN_GROUP=true     # In group chats, reply only when bot is mentioned
WHATSAPP_TARGET_FILE=                      # Optional; can switch via /file <filename>
WHATSAPP_UPLOADED_DOCS_DIR=./uploaded_docs
WHATSAPP_INGEST_OUTPUT_DIR=./output
WHATSAPP_RAG_WORKING_DIR=./rag_storage_by_whatsapp_file
WHATSAPP_RUNTIME_STATE_DIR=./rag_storage_whatsapp_runtime
WHATSAPP_STARTUP_NOTIFY_ENABLED=true
WHATSAPP_STARTUP_NOTIFY_MESSAGE=agent is ready.
WHATSAPP_STARTUP_NOTIFY_CHAT_ID=
```

3. Run WhatsApp backend:

```bash
python3 client/whatsapp/runtime.py
# or
sliderag-whatsapp
```

4. Start chatting. Supported commands:
- `/file <filename>` switch the active document
- `/status` inspect runtime connection/queue/model status

> If bridge shows repeated `Status: 408` / no QR code:
> 1) verify WhatsApp Web is reachable from your runtime network,
> 2) ensure bridge proxy config is effective in the same terminal process,
> 3) ensure bridge keeps running (do not close after QR).

### How to set `ALLOW_FROM` and `STARTUP_NOTIFY_CHAT_ID`

After you send one message in QQ/WeChat/WhatsApp, check runtime logs for a line like:

```text
Inbound message: chat_id=..., sender_id=...
```

Use the `sender_id` value for your allowlist and startup notification target.

## 🔗 Related Projects
| Project | Description | Link |
|---|---|---|
| **RAG-Anything** | All-in-One RAG Framework | [GitHub](https://github.com/HKUDS/RAG-Anything.git) |
| **nanobot** | Ultra-Lightweight Personal AI Assistant | [GitHub](https://github.com/HKUDS/nanobot.git) |

## :hugs: Citation
If you find this project useful, please cite:

```bibtex
@software{sliderag2026,
  title={SlideRAG: PPT-Centric Multimodal RAG for Study Preview and Exam Review},
  author={He Liu,Jiahao Zhang},
  year={2026},
  url={https://github.com/Hitlh/SlideRAG}
}
```
