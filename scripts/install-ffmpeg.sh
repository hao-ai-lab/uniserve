#!/bin/sh
# Build the static FFmpeg 8.1.2 that UniServe decodes reference media with,
# and install ffmpeg and ffprobe into <prefix>/bin (default /usr/local).
# Needs a C toolchain, nasm, zlib1g-dev, xz-utils and curl.
set -eu

prefix="${1:-/usr/local}"
version=8.1.2
sha256=464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$work"
curl -fsSLO "https://ffmpeg.org/releases/ffmpeg-$version.tar.xz"
echo "$sha256  ffmpeg-$version.tar.xz" | sha256sum -c -
tar -xf "ffmpeg-$version.tar.xz"
cd "ffmpeg-$version"
./configure --enable-static --disable-shared --extra-ldflags=-static \
  --pkg-config-flags=--static --disable-doc --disable-debug --disable-ffplay \
  --disable-autodetect --enable-version3 --enable-zlib
make -j"$(nproc)"
mkdir -p "$prefix/bin"
install -m 755 ffmpeg ffprobe "$prefix/bin/"
