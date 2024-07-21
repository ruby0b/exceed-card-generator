{
  description = "Generate eXceed cards from a CSV file";

  inputs.nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-24.05";

  outputs = inputs:
    let
      forAllSystems = function: inputs.nixpkgs.lib.genAttrs
        [ "x86_64-linux" "aarch64-linux" ]
        (system: function inputs.nixpkgs.legacyPackages.${system});

      pythonDeps = ps: with ps; [ pillow strictyaml ];

      pkg = { python3 }: python3.pkgs.buildPythonApplication {
        pname = "exceed-card-generator";
        version = "0.1.0";
        src = ./.;
        pyproject = true;
        build-system = [ python3.pkgs.poetry-core ];
        dependencies = pythonDeps python3.pkgs;
        postFixup = ''
          wrapProgram $out/bin/exceed-card-generator --set ASSETS ${./assets}
        '';
      };
    in
    {
      packages = forAllSystems (pkgs: rec {
        default = exceed-card-generator;
        exceed-card-generator = pkgs.callPackage pkg { };
      });

      overlays.default = prev: final: {
        exceed-card-generator = final.callPackage pkg { };
      };

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [ (pkgs.python3.withPackages pythonDeps) pkgs.poetry ];
        };
      });
    };
}
