{
  description = "Alpaca — Ollama-style vLLM wrapper with Ray-distributed inference";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { self
    , nixpkgs
    , flake-utils
    , pyproject-nix
    , uv2nix
    , pyproject-build-systems
    }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # vLLM/torch ship CUDA in their wheels, so we need unfree + CUDA support.
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        lib = nixpkgs.lib;

        python = pkgs.python312;

        # Load the uv workspace from this directory (pyproject.toml + uv.lock).
        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

        # Prefer prebuilt wheels: vLLM and torch have no usable sdist build here.
        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };

        # uv lock pins some packages whose wheels need their shared libs patched
        # for the Nix store (CUDA, etc.). autoPatchelf handles the common cases.
        pyprojectOverrides = final: prev: { };

        pythonSet =
          (pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
          }).overrideScope (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.default
              overlay
              pyprojectOverrides
            ]
          );

        # Default build: CLI + core deps, no GPU extra (resolves/builds GPU-free).
        coreEnv = pythonSet.mkVirtualEnv "alpaca-env" workspace.deps.default;

        # Full env including the `gpu` extra (vLLM + Ray) for GPU hosts.
        gpuEnv = pythonSet.mkVirtualEnv "alpaca-gpu-env"
          (workspace.deps.default // { alpaca-cli = [ "gpu" ]; });
      in
      {
        packages = {
          default = coreEnv;
          alpaca = coreEnv;
          alpaca-gpu = gpuEnv;
        };

        apps.default = {
          type = "app";
          program = "${coreEnv}/bin/alpaca";
        };

        devShells = {
          # Impure shell: uv manages a .venv itself. Best for iterating with
          # `uv sync` / `uv run alpaca` and for installing the GPU extra locally.
          default = pkgs.mkShell {
            packages = [
              python
              pkgs.uv
            ];
            env = {
              UV_PYTHON = python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
            };
            shellHook = ''
              unset PYTHONPATH
            '';
          };

          # Pure shell: editable install of alpaca against the Nix-built deps.
          # No uv resolution at runtime; matches CI exactly.
          nix = pkgs.mkShell {
            packages = [
              (pythonSet.mkVirtualEnv "alpaca-dev-env"
                (workspace.deps.all // { alpaca-cli = [ "gpu" ]; }))
              pkgs.uv
            ];
            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON_DOWNLOADS = "never";
            };
            shellHook = ''
              unset PYTHONPATH
            '';
          };
        };
      });
}
