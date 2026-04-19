#!/usr/bin/env bash
# Databricks cluster / Apps init script.
#
# Installs esmini + Xvfb + mesa software GL so region_envelope_injector
# can render 3D .xosc previews headlessly. Tested on Ubuntu 22.04 DBR 14+.
#
# Usage: attach as an Apps `setup.sh` or a cluster init-script.
set -euo pipefail

ESMINI_VERSION="${ESMINI_VERSION:-v3.0.2}"
ESMINI_ROOT="${ESMINI_ROOT:-/opt/esmini}"
ESMINI_ZIP_URL="https://github.com/esmini/esmini/releases/download/${ESMINI_VERSION}/esmini-bin_Linux.zip"

echo "[setup] installing system deps for esmini + Xvfb"
apt-get update -y >/dev/null
apt-get install -y --no-install-recommends \
    xvfb \
    libgl1-mesa-glx libgl1-mesa-dri libglu1-mesa \
    libxrandr2 libxinerama1 libxcursor1 libxi6 libxxf86vm1 \
    libfontconfig1 libsm6 \
    unzip wget ffmpeg >/dev/null

if [[ ! -x "${ESMINI_ROOT}/bin/esmini" ]]; then
    echo "[setup] fetching esmini ${ESMINI_VERSION}"
    mkdir -p "${ESMINI_ROOT}"
    tmp="$(mktemp -d)"
    wget -q -O "${tmp}/esmini.zip" "${ESMINI_ZIP_URL}"
    unzip -q "${tmp}/esmini.zip" -d "${tmp}"
    # Archive extracts to ./esmini/, move its contents into ESMINI_ROOT.
    src="${tmp}/esmini"
    [[ -d "${src}" ]] || src="${tmp}"
    cp -r "${src}/." "${ESMINI_ROOT}/"
    chmod +x "${ESMINI_ROOT}/bin/"* || true
fi

echo "[setup] esmini at ${ESMINI_ROOT}"
echo "[setup] done. Ensure env var ESMINI_ROOT=${ESMINI_ROOT} for the app."
