# Contributing to Explainium 2.0

Thanks for your interest in contributing! This document keeps it fast and predictable.

## Development Workflow
1. Fork + clone repo
2. Create a feature branch: `git checkout -b feature/short-description`
3. (Optional) Create / activate virtualenv
4. Install deps: `pip install -r requirements.txt`
5. Run quick health check: `./scripts/health_check.sh`
6. Make changes with clear commits
7. Add / update tests if logic changes
8. Open PR with concise description (WHAT + WHY)

## Pull Request Checklist
- [ ] Code passes basic lint (flake8/ruff if configured)
- [ ] README or doc updated if user-facing behavior changed
- [ ] No large unrelated formatting diffs
- [ ] Added tests for new functionality or edge cases
- [ ] No secrets / tokens committed

## Commit Message Style
```
feat: add video chunk truncation to respect ctx window
fix: prevent timeout loop when LLM disabled
chore: update dependency pins for stability
```

## Testing Guidelines
If adding extraction logic:
- Provide at least one sample input (redacted if proprietary)
- Assert entity count > 0 and confidence thresholds met

## Performance Expectations
- Keep single document processing under 120s target
- Avoid parallel llama calls (serialized executor)
- Use env flags for heavy toggles

## Local Models
Large model weights are never committed. Place them under `models/` as per README. `.gitignore` excludes them.

## Issue Reports
Include:
- Platform (OS, Python version, RAM)
- Exact command / action
- Stack trace (if any)
- Whether using LLM or pattern fallback mode

## Security
If you find a vulnerability, please open a minimal private report (or mark the issue clearly) — do not publish exploit details.

Happy contributing!
