# Seda (صدا)

Desktop app that clones your Persian voice and talks with local Ollama models
and Claude Code using it.

```
app/        Rust, gpui-kit desktop UI (binary: seda)
sidecar/    Python ML server (uv), started by the app on demand
packaging/  .desktop entry
```

See [SPEC.md](SPEC.md) for the design and [sidecar/README.md](sidecar/README.md)
for the models and their licenses.

## Requirements

- Linux (Wayland or X11), a Rust toolchain (`cargo`), `make`
- [`uv`](https://docs.astral.sh/uv/) on `PATH`: the app uses it to run the sidecar
- `ffmpeg`; an NVIDIA GPU with CUDA for OmniVoice and training (the `pocket_v2`
  engine also runs on the CPU)
- Optional: [Ollama](https://ollama.com) on `:11434`, Claude Code

## Install

```sh
make install
```

This builds the release binary as your user, then asks for `sudo` to copy the
files into `/usr/local`:

| What | Where |
|---|---|
| binary | `/usr/local/bin/seda` |
| sidecar sources | `/usr/local/share/seda/sidecar/` |
| launcher | `/usr/local/share/applications/seda.desktop` |
| icons | `/usr/local/share/icons/hicolor/*/apps/seda.{png,svg}` |

Seda then shows up in your app launcher. `sudo make install` also works: the
build still runs as your user so `app/target` never becomes root-owned.

The first time it starts the sidecar, `uv` creates the Python environment in
`~/.local/share/seda/sidecar-venv` (the installed sidecar directory is
read-only) and downloads PyTorch and the dependencies. Models are downloaded
from Hugging Face on first use. Its log is `~/.local/share/seda/sidecar.log`.

Options:

```sh
make install PREFIX=/usr                       # install somewhere else
make install DESTDIR=/tmp/stage                # staged install, no sudo (packaging)
make help                                      # all targets and variables
```

Use the same `PREFIX` for `make uninstall`. The sidecar path is built into the
binary, so change `PREFIX` by reinstalling, not by moving files.

## Uninstall

```sh
make uninstall
```

This removes the binary, the sidecar sources, the launcher and the icons (it asks
for `sudo`). Your data is kept: profiles, recordings, trained voices, chats and
settings stay in `~/.local/share/seda` (or `$SEDA_DATA`), together with the
sidecar's Python environment. To delete those as well:

```sh
rm -rf ~/.local/share/seda
```

Downloaded models stay in the Hugging Face cache (`~/.cache/huggingface`).

## Development

```sh
cd app && cargo run --release        # the app; it starts the sidecar from ./sidecar
cd sidecar && uv run seda-sidecar    # or run the sidecar yourself
```
