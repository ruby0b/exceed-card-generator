{
  description = "Generate eXceed cards from a CSV file";

  inputs.nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-24.05";

  outputs = { self, nixpkgs }:
    let
      forAllSystems = function: nixpkgs.lib.genAttrs
        [ "x86_64-linux" "aarch64-linux" ]
        (system: function nixpkgs.legacyPackages.${system});
      binFor = pkgs: pkgs.writers.writePython3Bin
        "exceed-card-generator"
        { libraries = with pkgs.python3.pkgs; [ pillow ]; }
        ("__import__('os').environ.setdefault('ASSETS', '${./assets}')\n"
          + builtins.readFile ./main.py);
    in
    {
      packages = forAllSystems (pkgs: {
        default = binFor pkgs;
        exceed-card-generator = binFor pkgs;
      });

      overlays.default = prev: final: {
        exceed-card-generator = binFor final;
      };

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [
            (pkgs.python3.withPackages (ps: with ps; [
              pillow
            ]))
          ];
        };
      });
    };
}
