# Agent review packets

These packets reconcile immutable legacy candidates with agent-prepared review
candidates. They reduce primary-review search and transcription, but are not human
annotation logs or evidence.

Packets validate against `schemas/ipke_review_packet.schema.json`. Artifact-qualified
references such as `legacy:C2` and `review:C2` prevent identifier collisions between
candidate versions. Transformation counts describe the agent-prepared rewrite only and
must not be reported as human effort or correction rates.
