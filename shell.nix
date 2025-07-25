{ nixpkgs ? import <nixpkgs> {} }:

nixpkgs.mkShell {
  nativeBuildInputs = with nixpkgs; [
    python310
  ];

  LD_LIBRARY_PATH = "${nixpkgs.stdenv.cc.cc.lib}/lib";
}
