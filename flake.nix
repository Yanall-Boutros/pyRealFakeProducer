{
  description = "Application packaged using poetry2nix";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable-small";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
        pkgs = nixpkgs.legacyPackages.${system};
        inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryApplication;
      in
      {
        packages = {
          myapp = mkPoetryApplication { projectDir = ./.; };
          default = self.packages.${system}.myapp;
        };

        # Shell for app dependencies.
        #
        #     nix develop
        #
        # Use this shell for developing your app.
        devShells.default = pkgs.mkShell {
          shellHook = ''
            export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
            export LD_LIBRARY_PATH=${pkgs.cudaPackages.cuda_nvrtc}/lib
            export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
            export EXTRA_CCFLAGS="-I/usr/include"
          '';
          packages = with pkgs; [
            linuxPackages.nvidia_x11
            libtorch-bin
            cudaPackages.cuda_nvrtc
            cudaPackages.cudnn
            python311Packages.torch-bin
            python311Packages.torchaudio-bin
            python311Packages.progressbar
            python311Packages.einops
            python311Packages.librosa
            python311Packages.unidecode
                (python311Packages.buildPythonPackage rec {
                   pname = "pyRealParser";
                   version = "0.1.0";
                   src = python311Packages.fetchPypi {
                     inherit pname version;
                     sha256="8a532dabb21909dc82d68fa82ad9bfb95ebc3f1d6a3c8398616d87081415fca4";                   
                   };
                })
            python311Packages.inflect
            python311Packages.rotary-embedding-torch
            python311Packages.safetensors
            python311Packages.transformers
          ];
          inputsFrom = [ 
            pkgs.python311
            pkgs.python311Packages.pip
            pkgs.libtorch-bin
            pkgs.python311Packages.torch-bin
            pkgs.python311Packages.torchaudio-bin
            pkgs.cudaPackages.cudnn
            pkgs.cudaPackages.cuda_nvrtc
            pkgs.cudaPackages.libcusparse
            self.packages.${system}.myapp
          ];
        };

        # Shell for poetry.
        #
        #     nix develop .#poetry
        #
        # Use this shell for changes to pyproject.toml and poetry.lock.
        devShells.poetry = pkgs.mkShell {
          packages = [ pkgs.poetry ];
        };
      });
}
