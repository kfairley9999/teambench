# Security policy

## This repository

TeamBench is a benchmark harness. The default path is an offline scripted mock. A live run talks to the Gemini API only when you pass `--model gemini` and set `GEMINI_API_KEY` in the environment.

- Do not commit API keys, `.env` files, or result artifacts that include request headers or secrets.
- Do not paste secrets into task fixtures, notes, or report files.
- The injected-output tasks (`T16`–`T18`) contain *fictional* bait strings such as `tok_live_abc`. They are not real credentials.
- CI never reads `GEMINI_API_KEY`.

## Reporting a vulnerability

Email Kevin Fairley at kfairley9999@gmail.com. Please do not open a public issue for a secret leak or an injection path that escapes the sandbox.

The tool sandbox is in-process and deterministic. It is not a security boundary against a malicious local plugin. Treat untrusted task packs as untrusted code.
