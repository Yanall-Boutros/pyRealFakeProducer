NIXPKGS_ALLOW_UNFREE=1 nix develop . --impure --show-trace --option substituters "https://cache.nixos-cuda.org https://cuda-maintainers.cachix.org" --option trusted-public-keys "cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M= cuda-maintainers.cachix.org-1:0dq3bveD76vTu74BsK75SBgdBJuCN3hb4hq87Sh67Y="

