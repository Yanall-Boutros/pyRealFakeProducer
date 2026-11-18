{ pkgs ? import <nixpkgs> {} }:

let
  melospy-unpacked = pkgs.stdenv.mkDerivation {
    pname = "melospy-gui-unpacked";
    version = "1.4.2";
    src = builtins.path {
      path = "${toString ./.}/melospy-gui_1.4.2~focal.deb";
      name = "melospy-gui-1.4.2-focal.deb";
    };

    nativeBuildInputs = [ pkgs.dpkg ];

    unpackPhase = ''
      dpkg-deb -x $src .
    '';

    installPhase = ''
      mkdir -p $out/bin $out/etc $out/share
      cp usr/bin/mss_gui $out/bin/
      cp -r etc/melospy-gui $out/etc/
      cp -r usr/share/* $out/share/

      # Point dirFeatures to the Nix store instead of /etc/melospy-gui
      substituteInPlace $out/etc/melospy-gui/config.yml \
        --replace-fail "/etc/melospy-gui/feature_definitions" "$out/etc/melospy-gui/feature_definitions"

      cat > $out/bin/melospy-gui <<EOF
#!/bin/sh
export QT_QPA_PLATFORM=xcb
export XKB_CONFIG_ROOT="${pkgs.xkeyboard_config}/share/X11/xkb"
exec $out/bin/mss_gui $out/etc/melospy-gui/config.yml "\$@"
EOF
      chmod +x $out/bin/melospy-gui
    '';
  };
in
pkgs.buildFHSEnv {
  name = "melospy-gui";

  targetPkgs = pkgs: with pkgs; [
    zlib
    glib
    dbus
    fontconfig
    freetype
    libGL
    libxkbcommon
    xkeyboard_config
    xorg.libX11
    xorg.libXext
    xorg.libXrender
    xorg.libxcb
    xorg.xcbutilwm
    xorg.xcbutilimage
    xorg.xcbutilkeysyms
    xorg.xcbutilrenderutil
    xorg.xcbutilcursor
    alsa-lib
    pulseaudio
  ];

  runScript = "${melospy-unpacked}/bin/melospy-gui";

  extraInstallCommands = ''
    mkdir -p $out/share
    cp -r ${melospy-unpacked}/share/* $out/share/
    substituteInPlace $out/share/applications/melospy-gui.desktop \
      --replace-warn "Exec=melospy-gui" "Exec=$out/bin/melospy-gui"
  '';
}

