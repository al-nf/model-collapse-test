{
  pkgs,
  lib,
  config,
  ...
}:
{
  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    venv.enable = true;
    venv.requirements = pkgs.lib.mkForce ''
      torch
      torchvision
      pillow
      numpy
    '';
  };

  # https://devenv.sh/packages/
  packages = [
    pkgs.nushell
  ];

  # See full reference at https://devenv.sh/reference/options/
}

