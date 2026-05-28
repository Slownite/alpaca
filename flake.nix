{
  Description = "Alpaca";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    flake-compat = "github:edolstra/flake-compat";
  };

  outputs = { self, nixpkgs, flake-utils, flake-compat }: flake-utils.lib.eachDefaultSystem (system: let
    pkgs = import nixpkgs { inherit system; };
    lib = nixpkgs.lib;

    devShell = pkgs.mkShell {
        packages = [
            pkgs.python313
            pkgs.python313Packages.uv
        ];
    };

    packages.default = pkgs.python313Packages.buildPythonApplication {
      pname = "alpaca";
      version = "0.1.0";
      src = ./.;
      propagatedBuildInputs = [];
    }; 
  in {
    packages.default = self.packages.${system}.default;
    devShell = self.devShell.${system};
  });
}