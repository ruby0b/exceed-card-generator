{
  description = "Generate eXceed cards from a CSV file";

  inputs.nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-24.05";

  outputs = inputs:
    let
      forAllSystems = function: inputs.nixpkgs.lib.genAttrs
        [ "x86_64-linux" "aarch64-linux" ]
        (system: function inputs.nixpkgs.legacyPackages.${system});
      binFor = pkgs: pkgs.writers.writePython3Bin
        "exceed-card-generator"
        { libraries = with pkgs.python3.pkgs; [ pillow strictyaml ]; }
        ("__import__('os').environ.setdefault('ASSETS', '${./assets}')\n"
          + builtins.readFile ./exceed_card_generator/main.py);
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
          packages = with pkgs; [
            (python3.withPackages (ps: with ps; [
              pillow
              strictyaml
            ]))
            uv
          ];
        };
      });
    };
}
