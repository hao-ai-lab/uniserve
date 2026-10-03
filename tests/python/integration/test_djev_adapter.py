"""DJev envelope behavior across the adapter's external HTTP boundary."""

import asyncio
import copy

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from examples.diffusion_gemma.djev_adapter import application

pytestmark = pytest.mark.integration


async def _decisions_preserve_states_images_and_candidate_probabilities():
    states = [
        {
            "id": "first",
            "state": {"door": "open"},
            "questions": {
                "open": {
                    "type": "boolean",
                    "instructions": "Is the door open?",
                    "criteria": {"true": "Open", "false": "Closed"},
                },
                "exit": {
                    "type": "choice",
                    "instructions": "Choose an exit.",
                    "criteria": {"north": "North", "east": "East"},
                },
                "risk": {
                    "type": "score",
                    "instructions": "Rate the risk.",
                    "criteria": ["Low", "High"],
                },
            },
            "images": ["data:image/png;base64,aW1hZ2U="],
        }
    ]
    second = copy.deepcopy(states[0])
    second["id"] = "second"
    second["state"] = "Another scene"
    states.append(second)

    async def evaluate(request):
        body = await request.json()
        # The stub is an external System One endpoint, enforcing its public
        # input contract rather than an internal adapter collaboration.
        assert body["model"] == "diffusiongemma"
        assert body["state"] in (states[0]["state"], states[1]["state"])
        assert body["x_images"] == states[0]["images"]
        expected = copy.deepcopy(states[0]["questions"])
        expected["open"]["type"] = "noul"
        assert body["questions"] == expected
        return web.json_response(
            {
                "answers": {
                    "open": {
                        "type": "noul",
                        "noul": 0.75,
                        "x_candidate_mass": 0.8,
                    },
                    "exit": {
                        "type": "choice",
                        "choice": "east",
                        "probabilities": {"north": 0.25, "east": 0.75},
                        "x_candidate_mass": 0.9,
                    },
                    "risk": {
                        "type": "score",
                        "score": 0.6,
                        "probabilities": {"0": 0.4, "1": 0.6},
                        "x_candidate_mass": 0.7,
                    },
                },
                "usage": {"input_tokens": 321},
            }
        )

    upstream = web.Application()
    upstream.router.add_post("/v1/systemone", evaluate)
    async with TestServer(upstream) as server:
        async with TestClient(
            TestServer(application(str(server.make_url("/")), "diffusiongemma"))
        ) as client:
            response = await client.post(
                "/api/evaluate", json={"states": states}
            )
            assert response.status == 200
            result = await response.json()
    assert [value["id"] for value in result["states"]] == ["first", "second"]
    for original, returned in zip(states, result["states"], strict=True):
        assert returned["state"] == original["state"]
        assert returned["questions"] == original["questions"]
        assert returned["prompt_tokens"] == 321
        assert returned["images"] == 1
        assert returned["answers"] == {
            "open": {
                "type": "boolean",
                "candidate_mass": 0.8,
                "probabilities": {"false": 0.25, "true": 0.75},
                "p_true": 0.75,
                "value": True,
            },
            "exit": {
                "type": "choice",
                "candidate_mass": 0.9,
                "probabilities": {"north": 0.25, "east": 0.75},
                "choice": "east",
                "value": "east",
            },
            "risk": {
                "type": "score",
                "candidate_mass": 0.7,
                "probabilities": {"0": 0.4, "1": 0.6},
                "score": 0.6,
                "level": 1,
                "value": 0.6,
            },
        }


async def _validation_and_capacity_failures_are_visible(
    upstream_status, expected
):
    async def evaluate(_request):
        return web.json_response(
            {"detail": "unavailable"}, status=upstream_status
        )

    upstream = web.Application()
    upstream.router.add_post("/v1/systemone", evaluate)
    state = {
        "id": "state",
        "state": "scene",
        "questions": {"q": {"type": "boolean", "instructions": "Open?"}},
    }
    async with TestServer(upstream) as server:
        async with TestClient(
            TestServer(application(str(server.make_url("/")), "diffusiongemma"))
        ) as client:
            response = await client.post(
                "/api/evaluate", json={"states": [state]}
            )
            assert response.status == expected
            assert (await response.json())["detail"] == {
                "detail": "unavailable"
            }
            invalid = await client.post(
                "/api/evaluate", json={"states": [state, state]}
            )
            assert invalid.status == 400
            options = await client.post(
                "/api/evaluate",
                json={"states": [state], "options": {"steps": 4}},
            )
            assert options.status == 400


def test_decisions_preserve_states_images_and_candidate_probabilities():
    asyncio.run(_decisions_preserve_states_images_and_candidate_probabilities())


@pytest.mark.parametrize(
    ("upstream_status", "expected"), ((422, 400), (503, 503))
)
def test_validation_and_capacity_failures_are_visible(
    upstream_status, expected
):
    asyncio.run(
        _validation_and_capacity_failures_are_visible(upstream_status, expected)
    )
