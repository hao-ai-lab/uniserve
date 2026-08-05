from __future__ import annotations

from typing import Any

from .base import BenchmarkTask, TaskRequest, apply_chat_image_contract, uses_reference_protocol


class T2ITask(BenchmarkTask):
    """Text-to-image requests through one declared wire shape."""

    def build_request(self, item: dict[str, Any]) -> TaskRequest:
        width = item.get("width", self.spec.width)
        height = item.get("height", self.spec.height)
        steps = item.get("steps", self.spec.steps)
        seed = item.get("seed", self.spec.seed)

        if self.spec.wire == "openai_chat_json":
            image_config: dict[str, Any] = {}
            if width is not None and height is not None:
                image_config["width"] = int(width)
                image_config["height"] = int(height)
            if steps is not None:
                image_config["steps"] = int(steps)
            if seed is not None:
                image_config["seed"] = int(seed)
            if self.spec.max_images is not None:
                image_config["num_images"] = int(self.spec.max_images)
            if self.spec.guidance_scale is not None:
                image_config["guidance_scale"] = self.spec.guidance_scale
            if self.spec.image_guidance_scale is not None:
                image_config["image_guidance_scale"] = self.spec.image_guidance_scale
            if self.spec.cfg_norm is not None:
                image_config["cfg_norm"] = self.spec.cfg_norm
            if self.spec.cfg_interval is not None:
                image_config["cfg_interval"] = list(self.spec.cfg_interval)
            if self.spec.timestep_shift is not None:
                image_config["timestep_shift"] = self.spec.timestep_shift
            root_parameters: dict[str, Any] = {}
            if self.spec.image_think is not None:
                root_parameters["think"] = self.spec.image_think
            if self.spec.image_t_eps is not None:
                root_parameters["t_eps"] = self.spec.image_t_eps
            payload: dict[str, Any] = {
                "model": self.spec.model,
                "modalities": ["image"],
                "messages": [{"role": "user", "content": item["prompt"]}],
                "temperature": self.spec.temperature,
                "top_p": self.spec.top_p,
                "ignore_eos": self.spec.ignore_eos,
            }
            apply_chat_image_contract(
                payload,
                image_config,
                root_parameters=root_parameters,
                include_reference_aliases=uses_reference_protocol(self.spec),
            )
            if self.spec.extra_request_body:
                payload.update(self.spec.extra_request_body)
            return TaskRequest(
                endpoint=self.spec.endpoint, payload=payload, kind="openai_chat_json"
            )

        payload = {"model": self.spec.model, "prompt": item["prompt"]}
        if width is not None and height is not None:
            payload["size"] = f"{width}x{height}"
        if steps is not None:
            payload["steps"] = int(steps)
        if seed is not None:
            payload["seed"] = int(seed)
        if self.spec.max_images is not None:
            payload["n"] = int(self.spec.max_images)
        if self.spec.guidance_scale is not None:
            payload["guidance_scale"] = self.spec.guidance_scale
        if self.spec.image_guidance_scale is not None:
            payload["image_guidance_scale"] = self.spec.image_guidance_scale
        if self.spec.cfg_norm is not None:
            payload["cfg_norm"] = self.spec.cfg_norm
        if self.spec.cfg_interval is not None:
            payload["cfg_interval"] = list(self.spec.cfg_interval)
        if self.spec.timestep_shift is not None:
            payload["timestep_shift"] = self.spec.timestep_shift
        if self.spec.extra_request_body:
            payload.update(self.spec.extra_request_body)
        return TaskRequest(endpoint=self.spec.endpoint, payload=payload, kind="images_generations")
