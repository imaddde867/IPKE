"""Locked IPKE-Bench annotation vocabulary."""

from __future__ import annotations

LOCKED_CONSTRAINT_TYPES: frozenset[str] = frozenset(
    {
        "precondition",
        "postcondition",
        "guard",
        "parameter",
        "role_assignment",
        "reference",
    }
)

LOCKED_ENFORCEMENT_LEVELS: frozenset[str] = frozenset({"must", "should", "may"})

CONSTRAINT_TYPE_INSTRUCTIONS = (
    "precondition, postcondition, guard, parameter, role_assignment, reference"
)

ENFORCEMENT_INSTRUCTIONS = "must, should, may"
