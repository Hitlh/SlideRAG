import argparse
import asyncio
import inspect
import os
import time
from typing import Any

from lightrag.llm.openai import openai_complete_if_cache
from openai import AsyncOpenAI


def _mask_secret(value: str | None) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def _build_messages(prompt: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": prompt}]


def _build_llm_model_func(api_key: str, base_url: str):
    def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return openai_complete_if_cache(
            kwargs.pop("model"),
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    return llm_model_func


async def _call_like_page_topic(
    llm_model_func: Any,
    model: str,
    prompt: str,
    max_tokens: int,
) -> str:
    try:
        result = llm_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            model=model,
            max_tokens=max_tokens,
            temperature=0,
        )
    except TypeError:
        result = llm_model_func(
            prompt,
            system_prompt=None,
            model=model,
            max_tokens=max_tokens,
            temperature=0,
        )

    if inspect.isawaitable(result):
        result = await result
    return str(result).strip()


async def _single_completion(
    client: AsyncOpenAI | None,
    backend: str,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    request_id: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        if backend == "page-topic":
            llm_model_func = _build_llm_model_func(api_key, base_url)
            content = await _call_like_page_topic(
                llm_model_func=llm_model_func,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            elapsed = time.perf_counter() - start
            return {
                "request_id": request_id,
                "ok": True,
                "elapsed": elapsed,
                "content": content,
                "usage": None,
            }

        if backend == "lightrag":
            response = openai_complete_if_cache(
                model,
                prompt,
                system_prompt=None,
                history_messages=[],
                api_key=api_key,
                base_url=base_url,
                max_tokens=max_tokens,
                temperature=0,
            )
            if inspect.isawaitable(response):
                response = await response
            elapsed = time.perf_counter() - start
            return {
                "request_id": request_id,
                "ok": True,
                "elapsed": elapsed,
                "content": str(response).strip(),
                "usage": None,
            }

        response = await client.chat.completions.create(
            model=model,
            messages=_build_messages(prompt),
            max_tokens=max_tokens,
            temperature=0,
        )
        elapsed = time.perf_counter() - start
        content = ""
        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content or ""
        return {
            "request_id": request_id,
            "ok": True,
            "elapsed": elapsed,
            "content": content.strip(),
            "usage": getattr(response, "usage", None),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            "request_id": request_id,
            "ok": False,
            "elapsed": elapsed,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


async def run_gateway_check(
    backend: str,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    rounds: int,
    timeout: float,
) -> int:
    client = None
    if backend == "sdk":
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=2,
        )

    print("Gateway config:")
    print(f"- backend: {backend}")
    print(f"- base_url: {base_url}")
    print(f"- model: {model}")
    print(f"- api_key: {_mask_secret(api_key)}")
    print(f"- timeout: {timeout}s")
    print(f"- concurrency: {concurrency}")
    print(f"- rounds: {rounds}")
    print()

    print("Step 1/2: single request smoke test")
    smoke = await _single_completion(
        client=client,
        backend=backend,
        api_key=api_key,
        base_url=base_url,
        model=model,
        prompt=f"{prompt}\n\nrequest_id=smoke-1",
        max_tokens=max_tokens,
        request_id="smoke-1",
    )
    if not smoke["ok"]:
        print(
            f"[FAIL] smoke request failed in {smoke['elapsed']:.2f}s "
            f"({smoke['error_type']}): {smoke['error']}"
        )
        if client is not None:
            await client.close()
        return 1
    print(f"[OK] smoke request finished in {smoke['elapsed']:.2f}s")
    print(f"Response preview: {smoke['content'][:120]}")
    print()

    print("Step 2/2: concurrent request test")
    all_results: list[dict[str, Any]] = []
    for round_idx in range(1, rounds + 1):
        tasks = [
            _single_completion(
                client=client,
                backend=backend,
                api_key=api_key,
                base_url=base_url,
                model=model,
                prompt=f"{prompt}\n\nrequest_index={i}\nround={round_idx}",
                max_tokens=max_tokens,
                request_id=f"round-{round_idx}-req-{i}",
            )
            for i in range(1, concurrency + 1)
        ]
        round_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        round_elapsed = time.perf_counter() - round_start
        all_results.extend(results)

        ok_count = sum(1 for item in results if item["ok"])
        fail_count = len(results) - ok_count
        print(
            f"Round {round_idx}: {ok_count}/{len(results)} succeeded, "
            f"{fail_count} failed, elapsed={round_elapsed:.2f}s"
        )
        for item in results:
            if item["ok"]:
                print(
                    f"  [OK] {item['request_id']} "
                    f"{item['elapsed']:.2f}s "
                    f"{item['content'][:80]}"
                )
            else:
                print(
                    f"  [FAIL] {item['request_id']} "
                    f"{item['elapsed']:.2f}s "
                    f"{item['error_type']}: {item['error']}"
                )
        print()

    if client is not None:
        await client.close()

    total_ok = sum(1 for item in all_results if item["ok"])
    total_fail = len(all_results) - total_ok
    print("Summary:")
    print(f"- total requests: {len(all_results)}")
    print(f"- success: {total_ok}")
    print(f"- failed: {total_fail}")

    if total_fail:
        error_groups: dict[str, int] = {}
        for item in all_results:
            if item["ok"]:
                continue
            key = item.get("error_type", "UnknownError")
            error_groups[key] = error_groups.get(key, 0) + 1
        print("- error groups:")
        for name, count in sorted(error_groups.items()):
            print(f"  - {name}: {count}")
        return 2

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an OpenAI-compatible gateway with single and concurrent chat completion calls.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="API key for the OpenAI-compatible gateway. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--backend",
        choices=["page-topic", "lightrag", "sdk"],
        default="page-topic",
        help="Call path to test. `page-topic` mirrors `_call_llm_model`; `lightrag` calls openai_complete_if_cache directly; `sdk` uses AsyncOpenAI directly.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", ""),
        help="Base URL for the OpenAI-compatible gateway. Defaults to OPENAI_BASE_URL.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="Chat model to test.",
    )
    parser.add_argument(
        "--prompt",
        default="Please reply with exactly: gateway_ok",
        help="Prompt used for the test requests.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Max tokens for each request.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="How many requests to send concurrently in each round.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="How many concurrent rounds to run.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY or pass --api-key.")
    if not args.base_url:
        raise SystemExit("Missing base URL. Set OPENAI_BASE_URL or pass --base-url.")
    my_prompt = """You receive the plain text extracted from one slide page. Return only JSON in this format:
{
  "page_idx": <int>,
  "topic": "<short topic string>"
}

Guidelines:
- Prefer existing page titles if they appear in the content.
- If no clear title, infer a 2-10 word noun phrase that best summarizes the page.
- Avoid explanations, numbering, or extra keys.
- If you cannot find a meaningful topic, fall back to: Disclaimer.

Page index: 2
Page text (truncated):
Disclaimer
This presentation contains forward looking statements which reflect Management's current views and estimates. The forward looking statements involve certain risks and uncertainties that could cause actual results to differ materially from those contained in the forward looking statements. Potential risks and uncertainties include such factors as general economic conditions, foreign exchange fluctuations, competitive product and pricing pressures and regulatory developments.

Language requirement: The topic must be in English."""
    exit_code = asyncio.run(
        run_gateway_check(
            backend=args.backend,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            prompt=my_prompt,
            max_tokens=args.max_tokens,
            concurrency=max(1, args.concurrency),
            rounds=max(1, args.rounds),
            timeout=max(1.0, args.timeout),
        )
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
