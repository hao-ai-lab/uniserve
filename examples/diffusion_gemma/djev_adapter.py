"""Serve DJev's /api/evaluate envelope through UniServe's System One API.

Install the repository's bench extra for aiohttp. Start UniServe separately,
then run this script with --upstream http://127.0.0.1:8000. This adapter maps
question and response fields; numerical readout settings belong to UniServe.
"""

from __future__ import annotations

import argparse
import asyncio
import time

from aiohttp import ClientError, ClientSession, ClientTimeout, web


def requests(payload, model):
    """Validate the multi-state envelope and map each state to System One.

    UniServe validates question bodies and image limits. DJev's unused
    ``options`` field is accepted only when empty, so settings cannot be
    silently ignored. State and question order are preserved.
    """
    if not isinstance(payload, dict) or set(payload) - {"states", "options"}:
        raise ValueError("expected a states object with optional empty options")
    if payload.get("options") not in (None, {}):
        raise ValueError("readout options must be configured on UniServe")
    states = payload.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError("states must be a nonempty list")
    seen, bodies = set(), []
    for state in states:
        if not isinstance(state, dict) or not {
            "id",
            "state",
            "questions",
        } <= set(state) <= {"id", "state", "questions", "images"}:
            raise ValueError("each state needs id, state and questions")
        identity = state["id"]
        if (
            not isinstance(identity, str)
            or not identity.strip()
            or identity in seen
        ):
            raise ValueError("state IDs must be unique nonempty strings")
        seen.add(identity)
        questions = state["questions"]
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a nonempty object")
        mapped = {}
        for name, question in questions.items():
            if not isinstance(question, dict) or question.get("type") not in (
                "boolean",
                "choice",
                "score",
            ):
                raise ValueError(
                    "question type must be boolean, choice or score"
                )
            mapped[name] = {
                **question,
                "type": "noul"
                if question["type"] == "boolean"
                else question["type"],
            }
        body = {"model": model, "state": state["state"], "questions": mapped}
        if "images" in state:
            body["x_images"] = state["images"]
        bodies.append(body)
    return states, bodies


def response_state(state, result):
    """Map one successful System One result into a DJev state result."""
    answers = {}
    for name, question in state["questions"].items():
        answer = result["answers"][name]
        kind = question["type"]
        value = {"type": kind, "candidate_mass": answer["x_candidate_mass"]}
        if kind == "boolean":
            probability = answer["noul"]
            value.update(
                probabilities={"false": 1.0 - probability, "true": probability},
                p_true=probability,
                value=probability >= 0.5,
            )
        elif kind == "choice":
            value.update(
                probabilities=answer["probabilities"],
                choice=answer["choice"],
                value=answer["choice"],
            )
        else:
            distribution = answer["probabilities"]
            value.update(
                probabilities=distribution,
                score=answer["score"],
                level=int(max(distribution, key=distribution.get)),
                value=answer["score"],
            )
        answers[name] = value
    return {
        "id": state["id"],
        "state": state["state"],
        "questions": state["questions"],
        "answers": answers,
        "prompt_tokens": result["usage"]["input_tokens"],
        "images": len(state.get("images", [])),
    }


def application(upstream, model):
    """Create an async adapter with one pooled client for its lifetime."""
    app = web.Application(client_max_size=160_000_000)
    upstream = upstream.rstrip("/")
    session = None

    async def client(_app):
        nonlocal session
        async with ClientSession(timeout=ClientTimeout(total=600)) as session:
            yield

    async def health(_request):
        try:
            async with session.get(f"{upstream}/health") as reply:
                ready = reply.status == 200
                return web.json_response(
                    {"ready": ready, "model": model},
                    status=200 if ready else 503,
                )
        except (ClientError, TimeoutError):
            return web.json_response(
                {"ready": False, "model": model}, status=503
            )

    async def post(body):
        async with session.post(f"{upstream}/v1/systemone", json=body) as reply:
            return reply.status, await reply.json()

    async def evaluate(request):
        started = time.perf_counter()
        try:
            states, bodies = requests(await request.json(), model)
        except (ValueError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)
        try:
            results = await asyncio.gather(*(post(body) for body in bodies))
        except (ClientError, TimeoutError) as error:
            return web.json_response({"error": str(error)}, status=502)
        for status, result in results:
            if status != 200:
                return web.json_response(
                    {"error": "UniServe request failed", "detail": result},
                    status=400 if status == 422 else status,
                )
        return web.json_response(
            {
                "model": model,
                "states": [
                    response_state(state, result)
                    for state, (_, result) in zip(states, results, strict=True)
                ],
                "execution": {
                    "states": len(states),
                    "questions": sum(
                        len(state["questions"]) for state in states
                    ),
                    "autoregressive_decode_steps": 0,
                    "network_model_calls": len(states),
                    "server_evaluation_seconds": time.perf_counter() - started,
                },
            }
        )

    app.cleanup_ctx.append(client)
    app.router.add_get("/api/health", health)
    app.router.add_post("/api/evaluate", evaluate)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="diffusiongemma")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    web.run_app(
        application(args.upstream, args.model), host=args.host, port=args.port
    )


if __name__ == "__main__":
    main()
