# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Scope of v0.1.0

- VLM providers: `host` (uses Claude Code / Cowork subscription) and `openrouter`
  (gateway to 200+ models via the `openai` SDK with custom `base_url`).
- Output modes: `separate`, `merged`, `all`.
- Output directory: always `./CLIPS/` in the user's cwd.

### Explicitly deferred

- Direct Anthropic / OpenAI / Gemini SDKs → v0.2.0
- CapCut export (`pyCapCut`) → v0.2.0 (pending upstream license clarification)
- Local providers (Ollama, LM Studio) → v0.3.0
- MCP server, GUI → v0.4.0+

## [0.1.0a1] — TBD

Initial alpha. Project scaffolding, security primitives, CLI skeleton.
