"""MiniMax-H3 reference workloads shared by the reference generators.

A workload fixes everything a reference run needs except the seed and the
implementation: the task, the prompt, the target canvas and duration, and
the condition media in request order. Media paths are relative to the inputs
directory written by ``fetch_inputs.py``; prompts are either the official
H3-Context-IR prompts of the three reproducible request scripts (read from
``prompts.json``) or prompts authored here for the condition layouts the
official scripts do not cover.

The official requests use seed 0; every workload also runs seed 42, the
default seed of the serving API.
"""

import dataclasses
import json
from pathlib import Path

OFFICIAL_SEED = 0
DEFAULT_SEED = 42

# The official media, as named by fetch_inputs.py.
OFFICIAL_KEYFRAME = "media/official_fl2va_0_image.png"
OFFICIAL_VIDEO = "media/official_ref2va_0_video.mp4"
OFFICIAL_AUDIO = "media/official_ref2va_1_audio.mp3"
SUBJECT_IMAGE = "media/hf_character_action_reference.png"
FOX_IMAGE = "media/hf_fl2va_clay_fox_reference.png"
ROBOT_VIDEO = "media/hf_robot_arm_red_cube.mp4"


@dataclasses.dataclass(frozen=True)
class Condition:
    """One request condition.

    Attributes:
        kind: ``image``, ``video`` (its soundtrack is conditioned on as well
            when the file carries one) or ``audio``.
        media: Path of the file relative to the inputs directory.
        role: ``keyframe`` or ``reference``.
        frame_index: ``0`` or ``-1`` for a keyframe, ``None`` otherwise.
    """

    kind: str
    media: str
    role: str = "reference"
    frame_index: int | None = None


@dataclasses.dataclass(frozen=True)
class Workload:
    """A fixed reference request.

    Attributes:
        name: Stable directory name of the workload.
        task: ``t2va``, ``fl2va`` or ``ref2va``.
        prompt: ``official:<task>`` for an official prompt, or the literal
            prompt text.
        aspect_ratio: ``W:H`` or ``auto``.
        duration_seconds: Requested duration; frames are ``round(d * 24)``
            rounded up to ``17 n + 5``.
        conditions: Conditions in request order.
        seeds: Seeds the workload runs with.
    """

    name: str
    task: str
    prompt: str
    aspect_ratio: str
    duration_seconds: float
    conditions: tuple[Condition, ...] = ()
    seeds: tuple[int, ...] = (DEFAULT_SEED,)

    def resolve_prompt(self, inputs: Path) -> str:
        """Return the prompt text, reading official prompts from inputs."""
        if self.prompt.startswith("official:"):
            prompts = json.loads((inputs / "prompts.json").read_text())
            return prompts[self.prompt.removeprefix("official:")]
        return self.prompt

    def request_body(self, inputs: Path, seed: int) -> dict:
        """Return the serving request body of this workload and seed."""
        conditions = []
        for condition in self.conditions:
            entry = {
                "type": condition.kind,
                "uri": f"file://{(inputs / condition.media).resolve()}",
                "role": condition.role,
            }
            if condition.frame_index is not None:
                entry["frame_index"] = condition.frame_index
            conditions.append(entry)
        return {
            "task": self.task,
            "prompt": self.resolve_prompt(inputs),
            "conditions": conditions,
            "target": {
                "short_edge": 768,
                "aspect_ratio": self.aspect_ratio,
                "duration_seconds": self.duration_seconds,
            },
            "seed": seed,
        }


_SUBJECT_PROMPT = (
    "subject_definitions:\n"
    "<Subject 1> is the young man with dark curly hair wearing a white "
    "T-shirt in <Picture 1>.\n\n"
    "summary:\n"
    "[subject reference] <Subject 1> washes a white bowl at a kitchen sink, "
    "then turns toward the window and smiles.\n\n"
    "integrated_multimodal_description: [Shot 1] Live-action, medium shot, "
    "static camera. In a warm, sunlit kitchen, <Subject 1> rinses a white "
    "ceramic bowl under running water, places it on the drying rack and "
    "glances toward the window with a relaxed smile.\n"
    "overall_soundscape: Running tap water, the soft clink of ceramic and "
    "faint birdsong through an open window.\n"
    "non_diegetic_music: None."
)

_SUBJECT_VOICE_PROMPT = (
    "subject_definitions:\n"
    "<Subject 1> is the young man with dark curly hair wearing a white "
    "T-shirt in <Picture 1>.\n"
    "<Audio 1> is the voice timbre reference for <Subject 1>'s voice.\n\n"
    "summary:\n"
    "[subject reference + audio reference] <Subject 1> stands in a kitchen "
    "and speaks to the camera in the voice of <Audio 1>.\n\n"
    "integrated_multimodal_description: [Shot 1] Live-action, medium "
    "close-up, slow push-in. In a sunlit kitchen, <Subject 1> dries his "
    "hands on a towel, looks into the camera and says, in the voice of "
    '<Audio 1>: "Dinner is almost ready, come sit down."\n'
    "overall_soundscape: Quiet kitchen room tone with a faint hum of a "
    "refrigerator.\n"
    "non_diegetic_music: None."
)

_TWO_VIDEOS_PROMPT = (
    "subject_definitions:\n"
    "<Subject 1> is the young man with short wavy blonde hair in a bright "
    "pink suit holding a small black lamb in <Video 1>.\n"
    "<Subject 2> is the white robotic arm with a black gripper in "
    "<Video 2>.\n"
    "<Audio 1> is the synchronized audio track of <Video 1>.\n"
    "<Audio 2> is the synchronized audio track of <Video 2>.\n\n"
    "summary:\n"
    "[multi-video reference] <Subject 1> stands on a grassy hill at golden "
    "hour while <Subject 2> places a red cube into a glass jar on a table "
    "beside him.\n\n"
    "integrated_multimodal_description: [Shot 1] Cinematic, medium wide "
    "shot, slow lateral dolly. On a grassy hillside at sunset, <Subject 1> "
    "cradles the black lamb while, on a white table next to him, "
    "<Subject 2> lifts a red cube and lowers it into a glass jar, following "
    "the motion of <Video 2>.\n"
    "overall_soundscape: Wind over the grass, distant sheep and the soft "
    "whir of the robotic arm's motors, as in <Audio 2>.\n"
    "non_diegetic_music: The gentle background music of <Audio 1>."
)

WORKLOADS = {
    workload.name: workload
    for workload in (
        Workload(
            name="t2va_16x9_5s",
            task="t2va",
            prompt="official:t2va",
            aspect_ratio="16:9",
            duration_seconds=5.0,
            seeds=(DEFAULT_SEED, OFFICIAL_SEED),
        ),
        Workload(
            name="t2va_9x16_5s",
            task="t2va",
            prompt="official:t2va",
            aspect_ratio="9:16",
            duration_seconds=5.0,
            seeds=(DEFAULT_SEED, OFFICIAL_SEED),
        ),
        # The shortest and longest durations serving admits: 107 and 362
        # frames. The diffusers pipeline rejects both by its duration check
        # alone, which the reference generator lifts.
        Workload(
            name="t2va_16x9_4s",
            task="t2va",
            prompt="official:t2va",
            aspect_ratio="16:9",
            duration_seconds=4.0,
        ),
        Workload(
            name="t2va_16x9_15s",
            task="t2va",
            prompt="official:t2va",
            aspect_ratio="16:9",
            duration_seconds=15.0,
        ),
        # The official FL2VA request: first keyframe, 8 s, auto canvas.
        Workload(
            name="fl2va_first_8s",
            task="fl2va",
            prompt="official:fl2va",
            aspect_ratio="auto",
            duration_seconds=8.0,
            conditions=(Condition("image", OFFICIAL_KEYFRAME, "keyframe", 0),),
            seeds=(DEFAULT_SEED, OFFICIAL_SEED),
        ),
        Workload(
            name="fl2va_last_8s",
            task="fl2va",
            prompt="official:fl2va",
            aspect_ratio="auto",
            duration_seconds=8.0,
            conditions=(Condition("image", OFFICIAL_KEYFRAME, "keyframe", -1),),
        ),
        # The 1376x768 last keyframe differs from the 1344x768 canvas the
        # first keyframe resolves, so it takes the cover-crop path.
        Workload(
            name="fl2va_first_last_8s",
            task="fl2va",
            prompt="official:fl2va",
            aspect_ratio="auto",
            duration_seconds=8.0,
            conditions=(
                Condition("image", OFFICIAL_KEYFRAME, "keyframe", 0),
                Condition("image", FOX_IMAGE, "keyframe", -1),
            ),
        ),
        Workload(
            name="ref2va_image_5s",
            task="ref2va",
            prompt=_SUBJECT_PROMPT,
            aspect_ratio="auto",
            duration_seconds=5.0,
            conditions=(Condition("image", SUBJECT_IMAGE),),
        ),
        Workload(
            name="ref2va_image_audio_5s",
            task="ref2va",
            prompt=_SUBJECT_VOICE_PROMPT,
            aspect_ratio="auto",
            duration_seconds=5.0,
            conditions=(
                Condition("image", SUBJECT_IMAGE),
                Condition("audio", OFFICIAL_AUDIO),
            ),
        ),
        # The official Ref2VA request: a video with its soundtrack and an
        # audio reference, 5 s, auto canvas.
        Workload(
            name="ref2va_video_audio_5s",
            task="ref2va",
            prompt="official:ref2va",
            aspect_ratio="auto",
            duration_seconds=5.0,
            conditions=(
                Condition("video", OFFICIAL_VIDEO),
                Condition("audio", OFFICIAL_AUDIO),
            ),
            seeds=(DEFAULT_SEED, OFFICIAL_SEED),
        ),
        Workload(
            name="ref2va_two_videos_5s",
            task="ref2va",
            prompt=_TWO_VIDEOS_PROMPT,
            aspect_ratio="auto",
            duration_seconds=5.0,
            conditions=(
                Condition("video", OFFICIAL_VIDEO),
                Condition("video", ROBOT_VIDEO),
            ),
        ),
        # A reference image plus a first keyframe. The diffusers integration
        # has no keyframe input for ref2va, so this layout has no diffusers
        # reference; SGLang's hybrid ref2va layout implements it.
        Workload(
            name="ref2va_image_keyframe_5s",
            task="ref2va",
            prompt=_SUBJECT_PROMPT,
            aspect_ratio="auto",
            duration_seconds=5.0,
            conditions=(
                Condition("image", SUBJECT_IMAGE),
                Condition("image", FOX_IMAGE, "keyframe", 0),
            ),
        ),
    )
}
