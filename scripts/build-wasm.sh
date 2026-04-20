#!/usr/bin/env bash
# build-wasm — produce the stq8-web WASM package for harmony-client.
#
# Output lands in stq8-web/pkg/ (gitignored). harmony-client expects a
# sibling clone at ../harmony-stq8/ and imports via its Vite alias, so
# re-run this after pulling stq8-core or stq8-web changes.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v wasm-pack >/dev/null 2>&1; then
  echo "error: wasm-pack not installed" >&2
  echo "  install: cargo install wasm-pack --locked" >&2
  echo "  or download: https://github.com/rustwasm/wasm-pack/releases" >&2
  exit 1
fi

wasm-pack build --target web stq8-web "$@"
