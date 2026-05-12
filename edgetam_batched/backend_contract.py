"""Strict backend contract checks for full batch=3 EdgeTAM multi-session work."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .config import BACKEND_BATCHED_MULTISESSION, BACKEND_BATCHED_MULTISESSION_TRT


class FullBatchedContractError(RuntimeError):
    """Raised when a backend claims full multi-session batching but does not meet the contract."""


@dataclass
class BackendContractResult:
    backend: str
    batch_vision: bool = False
    batch_memory_attention: bool = False
    batch_mask_decoder: bool = False
    batch_memory_encoder: bool = False
    batched_state_scatter: bool = False
    used_public_session_step_in_hot_path: bool = False
    partial_fallback_used: bool = False
    blockers: list[str] = field(default_factory=list)
    component_runtime: str | None = None
    trt_scope: str | None = None
    trt_memory_attention: bool = False
    trt_mask_decoder: bool = False
    trt_memory_encoder: bool = False
    torch_fallback_used: bool = False

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract_pass"] = self.contract_pass
        payload["missing_requirements"] = self.missing_requirements()
        return payload

    @property
    def contract_pass(self) -> bool:
        return not self.missing_requirements()

    def missing_requirements(self) -> list[str]:
        missing: list[str] = []
        if self.backend not in {BACKEND_BATCHED_MULTISESSION, BACKEND_BATCHED_MULTISESSION_TRT}:
            missing.append("backend is not a full batched multisession backend")
        for field_name in (
            "batch_vision",
            "batch_memory_attention",
            "batch_mask_decoder",
            "batch_memory_encoder",
            "batched_state_scatter",
        ):
            if not bool(getattr(self, field_name)):
                missing.append(field_name)
        if self.used_public_session_step_in_hot_path:
            missing.append("used_public_session_step_in_hot_path")
        if self.partial_fallback_used:
            missing.append("partial_fallback_used")
        if self.torch_fallback_used:
            missing.append("torch_fallback_used")
        return missing

    def assert_full_batched(self) -> None:
        assert_hf_batched_multisession_contract(self)


def assert_hf_batched_multisession_contract(result: BackendContractResult) -> None:
    missing = result.missing_requirements()
    if missing:
        details = "; ".join(missing)
        blockers = "; ".join(result.blockers)
        if blockers:
            details = f"{details}; blockers: {blockers}"
        raise FullBatchedContractError(f"hf_batched_multisession contract failed: {details}")


def contract_for_current_runtime(
    *,
    backend: str,
    batch_vision: bool,
    partial_fallback_used: bool,
    blockers: list[str] | None = None,
) -> BackendContractResult:
    """Return the explicit contract for the current implementation.

    The existing runtime can batch the vision encoder and then delegates to HF
    public session steps.  That is allowed for the diagnostic backend, but it is
    a hard contract failure for `hf_batched_multisession`.
    """

    blockers = list(blockers or [])
    used_public_step = True
    if backend == BACKEND_BATCHED_MULTISESSION:
        blockers.extend(
            [
                "memory attention is not batched across camera sessions",
                "mask decoder is not batched across camera sessions",
                "memory encoder/state update is not batched across camera sessions",
                "session state scatter is not implemented",
                "current code still calls model(inference_session=..., frame=...) per camera",
            ]
        )
    return BackendContractResult(
        backend=backend,
        batch_vision=batch_vision,
        batch_memory_attention=False,
        batch_mask_decoder=False,
        batch_memory_encoder=False,
        batched_state_scatter=False,
        used_public_session_step_in_hot_path=used_public_step,
        partial_fallback_used=partial_fallback_used,
        blockers=blockers,
    )


def render_contract_report(payload: dict[str, Any]) -> str:
    rows = [
        ["backend", payload["backend"]],
        ["contract_pass", payload["contract_pass"]],
        ["batch_vision", payload["batch_vision"]],
        ["batch_memory_attention", payload["batch_memory_attention"]],
        ["batch_mask_decoder", payload["batch_mask_decoder"]],
        ["batch_memory_encoder", payload["batch_memory_encoder"]],
        ["batched_state_scatter", payload["batched_state_scatter"]],
        ["used_public_session_step_in_hot_path", payload["used_public_session_step_in_hot_path"]],
        ["partial_fallback_used", payload["partial_fallback_used"]],
        ["missing_requirements", ", ".join(payload["missing_requirements"])],
    ]
    blocker_lines = "\n".join(f"- {item}" for item in payload.get("blockers", [])) or "- none"
    table = "\n".join(f"| {key} | {value} |" for key, value in rows)
    return "\n".join(
        [
            "# EdgeTAM Strict Full-Batched Backend Contract",
            "",
            "| field | value |",
            "| --- | --- |",
            table,
            "",
            "## Blockers",
            "",
            blocker_lines,
        ]
    )


def main() -> int:
    import argparse

    from .report_utils import write_json, write_markdown

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=BACKEND_BATCHED_MULTISESSION)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    result = contract_for_current_runtime(
        backend=args.backend,
        batch_vision=True,
        partial_fallback_used=True,
    )
    payload = result.to_json()
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render_contract_report(payload))
    return 0 if result.contract_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
