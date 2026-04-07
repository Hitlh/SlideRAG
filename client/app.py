import hashlib
import json
import os
import sys
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from client.utils import (
    find_available_port,
    get_env_bool,
    get_env_int,
    get_env_str,
    launch_streamlit_process,
    load_env_file,
    register_cleanup_handlers_once,
    run_async,
    wait_for_port_listening,
)

APP_ENTRY_PATH = Path(__file__).resolve()

load_env_file(PROJECT_ROOT / ".env")
register_cleanup_handlers_once()

# App config from environment
OPENAI_API_KEY = get_env_str("OPENAI_API_KEY", "")
OPENAI_BASE_URL = get_env_str("OPENAI_BASE_URL", "https://api.yunwu.ai/v1")

TEXT_LLM_MODEL = get_env_str("TEXT_LLM_MODEL", "gpt-4o-mini")
VLM_MODEL = get_env_str("VLM_MODEL", get_env_str("VISION_LLM_MODEL", "gpt-4o"))
AGENT_PROVIDER = get_env_str("AGENT_PROVIDER", "openai").strip().lower()
AGENT_MODEL = get_env_str("AGENT_MODEL", "gpt-4o")
ANTHROPIC_API_KEY = get_env_str("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = get_env_str("ANTHROPIC_BASE_URL", "")

EMBEDDING_MODEL = get_env_str("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIM = get_env_int("EMBEDDING_DIM", 3072)
EMBEDDING_MAX_TOKEN_SIZE = get_env_int("EMBEDDING_MAX_TOKEN_SIZE", 8192)

PARSER = get_env_str("PARSER", "mineru")
PARSE_METHOD = get_env_str("PARSE_METHOD", "auto")

ENABLE_IMAGE_PROCESSING = get_env_bool("ENABLE_IMAGE_PROCESSING", True)
ENABLE_TABLE_PROCESSING = get_env_bool("ENABLE_TABLE_PROCESSING", True)
ENABLE_EQUATION_PROCESSING = get_env_bool("ENABLE_EQUATION_PROCESSING", True)

RETRIEVE_TOP_K = get_env_int("RETRIEVE_TOP_K", 20)
RETRIEVE_CHUNK_TOP_K = get_env_int("RETRIEVE_CHUNK_TOP_K", 20)

OUTPUT_DIR = get_env_str("OUTPUT_DIR", "./output")
DOC_STORE_DIR = get_env_str("DOC_STORE_DIR", "./uploaded_docs")
WORKING_DIR_ROOT = get_env_str("WORKING_DIR_ROOT", "./rag_storage_by_file")
REGISTRY_PATH = get_env_str("REGISTRY_PATH", "./uploaded_docs_registry.json")

# ==========================================
# 🛠️ 1. Environment Check and Imports
# ==========================================
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.append(project_root_str)

try:
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc
    from raganything import RAGAnything, RAGAnythingConfig
    from rag_agent.agent.loop import AgentLoop
    from rag_agent.llm import AnthropicProvider, OpenAIProvider
except ImportError as e:
    st.error(
        f"❌ Critical error: Unable to import SlideRAG core libraries!\nPlease ensure app.py is located under the project root directory.\nDetails: {e}"
    )
    st.stop()

# ==========================================
# 🎨 2. Page Configuration
# ==========================================
st.set_page_config(
    page_title="SlideRAG (Single Doc Edition)",
    page_icon="📚",
    layout="wide",
)


# ==========================================
# 🧠 3. Core Engine Service Layer
# ==========================================
class RAGService:
    """
    RAG service wrapper.
    Manages RAGAnything instances to keep models resident in memory,
    avoiding reloads on each operation.
    """

    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.engine_pool = {}
        self.agent_loop_pool = {}

    def get_engine(self, working_dir):
        """Allow only one rag_instance per process; document switching is not allowed."""
        if working_dir in self.engine_pool:
            return self.engine_pool[working_dir]

        if self.engine_pool:
            existing_working_dir = next(iter(self.engine_pool.keys()))
            raise RuntimeError(
                "A rag_instance already exists in the current process; switching documents is not allowed. "
                f"Current cache directory: {existing_working_dir}; target cache directory: {working_dir}. "
                "To change files, restart the process and try again."
            )

        os.makedirs(working_dir, exist_ok=True)

        config = RAGAnythingConfig(
            working_dir=working_dir,
            parser=PARSER,
            parse_method=PARSE_METHOD,
            enable_image_processing=ENABLE_IMAGE_PROCESSING,
            enable_table_processing=ENABLE_TABLE_PROCESSING,
            enable_equation_processing=ENABLE_EQUATION_PROCESSING,
        )

        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                TEXT_LLM_MODEL,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=self.api_key,
                base_url=self.base_url,
                **kwargs,
            )

        embedding_func = EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
            send_dimensions=True,
            func=lambda texts, embedding_dim=None: openai_embed.func(
                texts,
                model=EMBEDDING_MODEL,
                api_key=self.api_key,
                base_url=self.base_url,
                embedding_dim=embedding_dim,
            ),
        )

        def vision_model_func(
            prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
        ):
            if messages:
                return openai_complete_if_cache(
                    VLM_MODEL,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    **kwargs,
                )
            if image_data:
                return openai_complete_if_cache(
                    VLM_MODEL,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt} if system_prompt else None,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                                },
                            ],
                        }
                        if image_data
                        else {"role": "user", "content": prompt},
                    ],
                    api_key=self.api_key,
                    base_url=self.base_url,
                    **kwargs,
                )
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        rag_instance = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )
        self.engine_pool[working_dir] = rag_instance
        return rag_instance

    def get_agent_loop(self, working_dir):
        """Reuse AgentLoop by working_dir to keep the same document session continuous."""
        if working_dir in self.agent_loop_pool:
            return self.agent_loop_pool[working_dir]

        rag_instance = self.get_engine(working_dir)
        provider = self._build_agent_provider()
        agent_workspace = os.path.join(working_dir, "agent_loop_workspace")
        os.makedirs(agent_workspace, exist_ok=True)

        loop = AgentLoop(
            provider=provider,
            workspace=agent_workspace,
            rag=rag_instance,
            model=AGENT_MODEL,
            retrieve_config={
                "mode": "hybrid",
                "top_k": RETRIEVE_TOP_K,
                "chunk_top_k": RETRIEVE_CHUNK_TOP_K,
            },
        )
        self.agent_loop_pool[working_dir] = loop
        return loop

    def _build_agent_provider(self):
        if AGENT_PROVIDER == "openai":
            if OpenAIProvider is None:
                raise RuntimeError("OpenAIProvider is unavailable. Please check rag_agent.llm imports.")
            return OpenAIProvider(
                api_key=self.api_key,
                api_base=self.base_url,
                default_model=AGENT_MODEL,
            )

        if AGENT_PROVIDER == "anthropic":
            if AnthropicProvider is None:
                raise RuntimeError(
                    "AnthropicProvider is unavailable. Please check anthropic dependencies and rag_agent.llm imports."
                )
            anthropic_key = ANTHROPIC_API_KEY or self.api_key
            anthropic_base = ANTHROPIC_BASE_URL or self.base_url
            return AnthropicProvider(
                api_key=anthropic_key,
                api_base=anthropic_base,
                default_model=AGENT_MODEL,
            )

        raise RuntimeError(
            f"Unsupported AGENT_PROVIDER: {AGENT_PROVIDER}. Please use 'openai' or 'anthropic'."
        )


# ==========================================
# 🖥️ 4. Frontend UI Logic
# ==========================================

if "rag_service" not in st.session_state:
    st.session_state.rag_service = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_indexed" not in st.session_state:
    st.session_state.doc_indexed = False
if "current_doc_name" not in st.session_state:
    st.session_state.current_doc_name = ""
if "current_working_dir" not in st.session_state:
    st.session_state.current_working_dir = ""
if "current_file_path" not in st.session_state:
    st.session_state.current_file_path = ""
if "post_parse_success_message" not in st.session_state:
    st.session_state.post_parse_success_message = ""

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".pptx"}
os.makedirs(DOC_STORE_DIR, exist_ok=True)
os.makedirs(WORKING_DIR_ROOT, exist_ok=True)


def load_registry():
    if not os.path.exists(REGISTRY_PATH):
        return {}
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_registry(registry):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def compute_file_sha256(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_working_dir_for_file(file_path):
    abs_path = str(Path(file_path).resolve())
    file_hash = compute_file_sha256(abs_path)
    working_dir = os.path.join(WORKING_DIR_ROOT, file_hash)

    registry = load_registry()
    registry[abs_path] = {
        "file_name": os.path.basename(abs_path),
        "file_hash": file_hash,
        "working_dir": working_dir,
        "updated_at": int(time.time()),
    }
    save_registry(registry)

    return file_hash, working_dir


with st.sidebar:
    st.title("📚 RAG Single-Document QA")
    st.caption("Single Document Parse + QA")
    st.divider()

    if OPENAI_API_KEY:
        st.caption("API Key: loaded from .env")
    else:
        st.caption("API Key: missing in .env")
    st.caption(f"Base URL (.env): {OPENAI_BASE_URL}")

    if st.session_state.post_parse_success_message:
        st.success(st.session_state.post_parse_success_message)
        st.session_state.post_parse_success_message = ""

    if st.button("🧩 Create New rag Instance"):
        try:
            new_port = find_available_port()
            with st.spinner("Starting a new process and waiting for service readiness..."):
                launch_streamlit_process(new_port, APP_ENTRY_PATH)
                ready = wait_for_port_listening(new_port, timeout_seconds=20.0)

            target = f"http://localhost:{new_port}"
            st.markdown(f"[Open New Process Page]({target})")
            if ready:
                st.success(f"New process started, port: {new_port}")
                components.html(
                    f"""
                                    <script>
                                        const target = '{target}';
                                        const opened = window.open(target, '_blank');
                                        if (!opened) {{
                                            try {{
                                                window.top.location.href = target;
                                            }} catch (e) {{
                                                // Keep manual link as fallback.
                                            }}
                                        }}
                                    </script>
                                    """,
                    height=0,
                )
            else:
                st.warning(
                    "The new process has started, but service initialization is slow. Please wait a few seconds and click the link above."
                )
        except Exception as e:
            st.error(f"Failed to start a new process: {str(e)}")

    st.divider()

    selected_file_path = None
    selected_file_name = None
    has_existing_engine = bool(
        st.session_state.rag_service and st.session_state.rag_service.engine_pool
    )

    if has_existing_engine:
        if st.session_state.current_doc_name:
            st.info(f"Current document: {st.session_state.current_doc_name}")
        selected_file_path = st.session_state.current_file_path or None
        selected_file_name = st.session_state.current_doc_name or ""
    else:
        source_mode = st.radio(
            "Document source",
            ["Upload new file", "Select existing file"],
            horizontal=True,
        )

        if source_mode == "Upload new file":
            uploaded_file = st.file_uploader(
                "📄 Upload a document for parsing",
                type=["pdf", "txt", "docx", "pptx"],
                accept_multiple_files=False,
            )

            if uploaded_file is not None:
                target_path = os.path.join(DOC_STORE_DIR, uploaded_file.name)

                new_bytes = uploaded_file.getvalue()
                if os.path.exists(target_path):
                    with open(target_path, "rb") as f:
                        old_bytes = f.read()
                    if old_bytes != new_bytes:
                        stem = Path(uploaded_file.name).stem
                        suffix = Path(uploaded_file.name).suffix
                        idx = 1
                        while True:
                            candidate = os.path.join(DOC_STORE_DIR, f"{stem}_{idx}{suffix}")
                            if not os.path.exists(candidate):
                                target_path = candidate
                                break
                            idx += 1

                if not os.path.exists(target_path):
                    with open(target_path, "wb") as f:
                        f.write(new_bytes)

                selected_file_path = target_path
                selected_file_name = os.path.basename(target_path)
                st.info(f"Stored at fixed location: {selected_file_name}")

        else:
            existing_files = sorted(
                [
                    f
                    for f in os.listdir(DOC_STORE_DIR)
                    if os.path.isfile(os.path.join(DOC_STORE_DIR, f))
                    and Path(f).suffix.lower() in ALLOWED_EXTENSIONS
                ]
            )

            if existing_files:
                selected_file_name = st.selectbox("📚 Select an uploaded document", existing_files)
                selected_file_path = os.path.join(DOC_STORE_DIR, selected_file_name)
                st.info(f"Using stored file: {selected_file_name}")

                confirm_delete = st.checkbox(
                    "Confirm deletion of the currently selected document", key="confirm_delete_existing_file"
                )
                if st.button("🗑️ Delete selected document"):
                    if not confirm_delete:
                        st.warning("Please check the confirmation box first.")
                    else:
                        try:
                            os.remove(selected_file_path)
                            abs_deleted_path = str(Path(selected_file_path).resolve())
                            registry = load_registry()
                            if abs_deleted_path in registry:
                                registry.pop(abs_deleted_path)
                                save_registry(registry)

                            if st.session_state.current_doc_name == selected_file_name:
                                st.session_state.doc_indexed = False
                                st.session_state.current_doc_name = ""
                                st.session_state.current_working_dir = ""
                                st.session_state.current_file_path = ""
                                st.session_state.messages = []
                            st.success(f"Deleted: {selected_file_name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Deletion failed: {str(e)}")
            else:
                st.warning("No available documents found in the fixed directory. Please upload a new file first.")

    if selected_file_path:
        _, planned_working_dir = resolve_working_dir_for_file(selected_file_path)
        st.caption(f"Cache directory: {planned_working_dir}")

        current_service = st.session_state.rag_service
        has_existing_engine = bool(current_service and current_service.engine_pool)
        existing_working_dir = ""
        if has_existing_engine:
            existing_working_dir = next(iter(current_service.engine_pool.keys()), "")
        switching_file_blocked = has_existing_engine and existing_working_dir != planned_working_dir
        if switching_file_blocked:
            st.warning(
                "A rag_instance already exists in the current process, so changing files is not allowed. "
                "Please restart the process before selecting a new file."
            )

        if st.button("🚀 Parse and inject into knowledge base", disabled=switching_file_blocked):
            if not OPENAI_API_KEY:
                st.error("OPENAI_API_KEY was not found. Please configure it in .env and try again.")
                st.stop()

            if st.session_state.rag_service is None:
                st.session_state.rag_service = RAGService(OPENAI_API_KEY, OPENAI_BASE_URL)

            _, working_dir = resolve_working_dir_for_file(selected_file_path)
            engine = st.session_state.rag_service.get_engine(working_dir)

            try:
                with st.spinner("Parsing document, please wait..."):
                    print(f"Start parsing document: {selected_file_path}")
                    run_async(
                        engine.process_document_complete_with_page_topics(
                            file_path=selected_file_path,
                            output_dir=OUTPUT_DIR,
                            parse_method=PARSE_METHOD,
                        )
                    )
                st.session_state.doc_indexed = True
                st.session_state.current_doc_name = selected_file_name
                st.session_state.current_working_dir = working_dir
                st.session_state.current_file_path = selected_file_path
                st.session_state.post_parse_success_message = (
                    f"✅ Parsing complete: {selected_file_name}. You can now start asking questions."
                )
                st.rerun()
            except Exception as e:
                st.session_state.doc_indexed = False
                st.error(f"Processing failed: {str(e)}")


st.subheader("💬 Knowledge Base Q&A")
if st.session_state.current_doc_name:
    st.caption(f"Current knowledge-base document: {st.session_state.current_doc_name}")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask a question based on the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if st.session_state.rag_service and st.session_state.doc_indexed:
        if not st.session_state.current_working_dir:
            st.error("The current document is not bound to a cache directory. Please re-parse the document first.")
            st.stop()

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                print(f"User question: {prompt}")
                agent_loop = st.session_state.rag_service.get_agent_loop(
                    st.session_state.current_working_dir
                )
                result = run_async(
                    agent_loop.process_message(
                        prompt,
                        file_path=st.session_state.current_file_path,
                        parse_method=PARSE_METHOD,
                    )
                )
                response = result.final_answer or ""
                st.write(response)
                if result.tools_used:
                    st.caption(f"Tools used: {', '.join(result.tools_used)}")
        st.session_state.messages.append({"role": "assistant", "content": str(response)})
    elif st.session_state.rag_service and not st.session_state.doc_indexed:
        st.error("Please upload and parse a document before starting Q&A.")
    else:
        st.error("Please initialize the engine from the left panel first.")
