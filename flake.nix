{
  description = "Minimal Nix Python environment with PyTorch and CUDA";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable-small";
  };

  nixConfig = {
    extra-substituters = [ "https://cache.nixos-cuda.org" ];
    extra-trusted-public-keys = [ "cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M=" ];
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          torch
          torchaudio
          progressbar
          einops
          pandas
          librosa
          unidecode
          inflect
          rotary-embedding-torch
          safetensors
          tokenizers
          (buildPythonPackage rec {
            pname = "pyRealParser";
            version = "0.1.0";
            pyproject = true;
            build-system = [ setuptools ];
            src = fetchPypi {
              inherit pname version;
              sha256 = "8a532dabb21909dc82d68fa82ad9bfb95ebc3f1d6a3c8398616d87081415fca4";
            };
            doCheck = false;
          })
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          shellHook = ''
            export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.cudaPackages.cuda_nvrtc
              pkgs.cudaPackages.cudnn
              pkgs.cudaPackages.libcusparse
              pkgs.cudaPackages.libcublas
            ]}
            export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
            export EXTRA_CCFLAGS="-I/usr/include"
          '';

          packages = [
            pythonEnv
            pkgs.cudaPackages.cuda_nvrtc
            pkgs.cudaPackages.cudnn
          ];
        };
      });
}
