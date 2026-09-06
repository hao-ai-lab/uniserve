"""H3 component assignments and unique media producers within one serving instance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ...bootstrap.plan import ComponentDeployConfig
from ...nn.mesh import DeviceMesh, GroupCoordinator


@dataclass(frozen=True)
class H3Placement:
    """Bind H3 component membership to process-local modules and stage products.

    Component configurations are the authoritative deployment input. Local
    meshes exist only for components assigned to this physical process. The
    process group belongs to the execution data plane and does not define any
    model-parallel degree.
    """

    components: Mapping[str, ComponentDeployConfig]
    meshes: Mapping[str, DeviceMesh]
    process_group: GroupCoordinator

    def __post_init__(self) -> None:
        expected = {"denoiser", "text_encoder", "video_decoder", "audio_decoder", "output"}
        if set(self.components) != expected:
            raise ValueError(f"H3 deployment must assign exactly {sorted(expected)}")
        for name, component in self.components.items():
            if any(rank not in self.process_group.ranks for rank in component.ranks):
                raise ValueError(f"H3 {name} members lie outside the serving instance")
            if name != "video_decoder" and component.distribution is not None:
                raise ValueError(
                    f"H3 {name} uses model-parallel membership, not temporal distribution"
                )
            if name in ("denoiser", "text_encoder"):
                mesh = self.meshes.get(name)
                if (mesh is not None) != self.owns(name):
                    raise ValueError(f"H3 {name} mesh must exist exactly on its assigned members")
                if mesh is not None and (
                    mesh.ranks != component.ranks
                    or mesh.parallel_config != component.parallel_config
                ):
                    raise ValueError(f"H3 {name} mesh disagrees with component deployment")
        denoiser = self.components["denoiser"].parallel_config
        if denoiser.pipeline_parallel_size > 50:
            raise ValueError("H3 pipeline stages cannot exceed its 50 transformer layers")
        if denoiser.sequence_parallel.kind not in {
            "local",
            "ulysses",
            "allgather",
            "ring",
            "hybrid",
            "attention2d",
        }:
            raise ValueError(
                "H3 context parallelism requires sparse global-selection and partial-attention support"
            )
        tensor = denoiser.tensor_parallel_size
        sequence = denoiser.sequence_parallel_size
        ulysses = dict(denoiser.dimensions)["ulysses"]
        if 56 % (tensor * ulysses) or 5376 % tensor or 14336 % tensor:
            raise ValueError(
                "H3 TP × Ulysses must divide 56 heads; TP must divide hidden and MLP widths"
            )
        if tensor not in (1, 2, 4) or sequence not in (1, 2, 4):
            raise ValueError("H3 resident execution requires TP and sequence degrees in 1, 2, or 4")
        encoder = self.components["text_encoder"].parallel_config
        if encoder.pipeline_parallel_size != 1 or encoder.sequence_parallel_size != 1:
            raise ValueError("H3 text encoder supports direct tensor parallelism")
        degree = encoder.tensor_parallel_size
        if any(width % degree for width in (64, 8, 25600)):
            raise ValueError(
                "H3 encoder TP must divide 64 query heads, 8 KV heads, and MLP width 25600"
            )
        video = self.components["video_decoder"]
        if video.distribution != "temporal_units" or video.units_per_rank != 1:
            raise ValueError("H3 video decoder requires temporal_units with native batch one")
        for name in ("audio_decoder", "output"):
            component = self.components[name]
            if len(component.ranks) != 1 or component.parallel_config.world_size != 1:
                raise ValueError(f"H3 {name} requires one local owner")

    def owns(self, component: str) -> bool:
        """Whether this process loads and executes the named component."""

        return self.process_group.rank in self.components[component].ranks

    @property
    def denoiser_mesh(self) -> DeviceMesh | None:
        return self.meshes.get("denoiser")

    @property
    def encoder_mesh(self) -> DeviceMesh | None:
        return self.meshes.get("text_encoder")

    @property
    def decoder_ranks(self) -> tuple[int, ...]:
        return self.components["video_decoder"].ranks

    @property
    def decoder_index(self) -> int | None:
        rank = self.process_group.rank
        return self.decoder_ranks.index(rank) if rank in self.decoder_ranks else None

    @property
    def audio_rank(self) -> int:
        return self.components["audio_decoder"].ranks[0]

    @property
    def output_rank(self) -> int:
        return self.components["output"].ranks[0]

    @property
    def input_ranks(self) -> tuple[int, ...]:
        """First-stage input owners across tensor and sequence coordinates."""

        component = self.components["denoiser"]
        stage_width = (
            component.parallel_config.world_size // component.parallel_config.pipeline_parallel_size
        )
        return component.ranks[:stage_width]

    @property
    def latent_producers(self) -> tuple[int, ...]:
        """Final-stage sequence-row owners, excluding tensor-replicated copies."""

        component = self.components["denoiser"]
        geometry = DeviceMesh(
            component.ranks,
            component.ranks[0],
            component.parallel_config,
            self.process_group.device,
        )
        tensor_axis = tuple(name for name, _ in geometry.dimensions).index("tp")
        pipeline_axis = tuple(name for name, _ in geometry.dimensions).index("pp")
        return tuple(
            rank
            for rank in component.ranks
            if geometry.get_coordinate(rank)[tensor_axis] == 0
            and geometry.get_coordinate(rank)[pipeline_axis]
            == component.parallel_config.pipeline_parallel_size - 1
        )
