"""Generation and cancellation through an independently launched AFD server.

Set UNISERVE_AFD_URL and UNISERVE_AFD_MODEL for the running deployment. The
deployment fixes its role placement, precision and graph policy before these
client-side checks; tests never change them or start another measurement.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


@pytest.fixture(scope="module")
def served():
    url = os.environ.get("UNISERVE_AFD_URL")
    model = os.environ.get("UNISERVE_AFD_MODEL")
    if not url or not model:
        pytest.fail("UNISERVE_AFD_URL and UNISERVE_AFD_MODEL are required")
    return url.rstrip("/"), model


def _request(model, prompt, limit=24):
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_completion_tokens": limit,
        "return_token_ids": True,
    }


def _post(served, prompt, limit=24):
    url, model = served
    response = httpx.post(
        f"{url}/v1/chat/completions",
        json=_request(model, prompt, limit),
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def _secret(served, code):
    result = _post(
        served, f"The secret code is {code}. Reply with only the secret code."
    )
    assert result["choices"][0]["message"]["content"].strip() == code, result
    assert result["choices"][0]["finish_reason"] == "stop"
    return result


def test_concurrent_requests_keep_their_own_generation(served):
    with ThreadPoolExecutor(4) as pool:
        results = list(
            pool.map(
                lambda code: _secret(served, code),
                ("violet", "314159", "orchid", "paperclip"),
            )
        )
    assert len({result["id"] for result in results}) == len(results)


def test_generation_obeys_its_token_limit(served):
    result = _post(served, "Count from one to one hundred, in words.", limit=3)
    assert 1 <= result["usage"]["completion_tokens"] <= 3
    assert result["choices"][0]["finish_reason"] in {"length", "stop"}


def test_cancelled_streams_release_capacity_for_later_requests(served):
    url, model = served
    for _ in range(3):
        body = _request(
            model, "Write the numbers from 1 to 1000 in order.", limit=256
        )
        body["stream"] = True
        with httpx.stream(
            "POST", f"{url}/v1/chat/completions", json=body, timeout=180
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if any(
                    choice.get("delta", {}).get("content")
                    for choice in chunk.get("choices", ())
                ):
                    break
            else:
                pytest.fail("the stream ended before emitting a token")
        _secret(served, "reuse")

    # More new admissions than cancelled streams expose retained request
    # capacity while also exercising joins from temporarily idle peers.
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: _secret(served, "ready"), range(8)))
