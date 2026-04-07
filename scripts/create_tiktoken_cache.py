import os

import tiktoken

# Directory where tiktoken cache files are stored.
cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR", "./temp/tiktoken")
os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir

os.makedirs(cache_dir, exist_ok=True)

print(f"Using TIKTOKEN_CACHE_DIR={cache_dir}")
print("Downloading and caching tiktoken models...")

# LightRAG default tokenizer path usually resolves to this encoding.
try:
	tiktoken.get_encoding("cl100k_base")
except Exception as exc:
	print("Failed to download tiktoken cache from network.")
	print(f"Error: {exc}")
	print("You can generate cache on another machine and copy it to this directory.")
	raise SystemExit(1) from exc

print(f"tiktoken models have been cached in '{cache_dir}'")
