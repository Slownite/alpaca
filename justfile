# Alpaca task runner. Run `just` to list recipes.

# Show available recipes
default:
    @just --list

# Sync the dev environment (creates/updates .venv from uv.lock)
sync:
    uv sync

# Sync including the GPU extra (vLLM + Ray)
sync-gpu:
    uv sync --extra gpu

# Run the alpaca CLI; forward any args, e.g. `just cli pull mistral`
cli *args:
    uv run alpaca {{args}}

# Print CLI help
help:
    uv run alpaca --help

# Run the test suite; forward pytest args, e.g. `just test -k cli`
test *args:
    uv run pytest {{args}}

# Build the wheel and sdist into ./dist
build:
    uv build

# Remove build artifacts and caches
clean:
    rm -rf dist build *.egg-info alpaca_cli.egg-info
    find . -type d -name __pycache__ -prune -exec rm -rf {} +
    find . -type d -name .pytest_cache -prune -exec rm -rf {} +

# Count lines of Python source (excludes tests and .venv)
loc:
    @find alpaca -name '*.py' | xargs wc -l | tail -1

# Count lines of Python source including tests
loc-all:
    @find alpaca tests -name '*.py' | xargs wc -l | tail -1

# Per-file line counts, largest last
loc-files:
    @find alpaca tests -name '*.py' | xargs wc -l | sort -n

# Enter the impure uv dev shell
shell:
    nix develop

# Enter the pure Nix dev shell (Nix-built deps)
shell-nix:
    nix develop .#nix

# Build the default Nix package (CLI + core deps, GPU-free)
nix-build:
    nix build

# Run the CLI via the Nix app
nix-run *args:
    nix run . -- {{args}}
