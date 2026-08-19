"""The checkpoint allowlist must cover everything that lives in graph state.

Human review is an interrupt, an interrupt is a durable write of the whole
state, and LangGraph will refuse to rebuild types it was not told about. That
failure is silent and only shows up on *resume* — so it is worth a structural
test rather than trusting a checklist in a docstring.
"""

import enum
import typing

from pydantic import BaseModel

from src.checkpointing import CHECKPOINTED_TYPES, serializer
from src.state import DueDiligenceState


def _project_types(annotation, seen=None) -> set[type]:
    """Every project-owned model/enum reachable from *annotation*."""
    seen = seen if seen is not None else set()

    for arg in typing.get_args(annotation):
        _project_types(arg, seen)

    origin = typing.get_origin(annotation)
    if origin is not None and not typing.get_args(annotation):
        return seen

    if not isinstance(annotation, type):
        return seen
    if not annotation.__module__.startswith("src."):
        return seen
    if annotation in seen:
        return seen

    if issubclass(annotation, (BaseModel, enum.Enum)):
        seen.add(annotation)
        if issubclass(annotation, BaseModel):
            for field in annotation.model_fields.values():
                _project_types(field.annotation, seen)

    return seen


class TestCheckpointAllowlist:
    def test_every_state_type_is_registered(self):
        """Add a Pydantic model to the state and forget src/checkpointing.py,
        and this fails here rather than when someone resumes a run."""
        reachable: set[type] = set()
        for annotation in typing.get_type_hints(
            DueDiligenceState, include_extras=True
        ).values():
            _project_types(annotation, reachable)

        missing = reachable - set(CHECKPOINTED_TYPES)
        assert not missing, (
            "these types travel in graph state but are not in "
            f"CHECKPOINTED_TYPES: {sorted(t.__name__ for t in missing)}"
        )

    def test_allowlist_has_no_dead_entries(self):
        """A stale entry is harmless but misleading about what state holds."""
        reachable: set[type] = set()
        for annotation in typing.get_type_hints(
            DueDiligenceState, include_extras=True
        ).values():
            _project_types(annotation, reachable)

        # PlanDecision and Classification cross the interrupt boundary rather
        # than living in the state schema, so they are legitimately extra.
        boundary_types = {"PlanDecision", "Classification"}
        dead = {
            t for t in CHECKPOINTED_TYPES
            if t not in reachable and t.__name__ not in boundary_types
        }
        assert not dead, f"unreachable allowlist entries: {sorted(t.__name__ for t in dead)}"

    def test_serializer_declares_an_explicit_allowlist(self):
        """Not the permissive default: unknown types must not be rebuilt from
        a checkpoint store, which is a code-execution path."""
        allowed = serializer()._allowed_msgpack_modules
        assert allowed is not True, "serializer is still in permissive mode"
        assert ("src.claims.models", "Claim") in allowed
