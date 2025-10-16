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
    venv.requirements = ./requirements.txt;
  };

  # https://devenv.sh/packages/
  packages = [
    pkgs.nushell
  ];

  processes = {
    gan.exec = "python gan.py";
  };

  # See full reference at https://devenv.sh/reference/options/
}

