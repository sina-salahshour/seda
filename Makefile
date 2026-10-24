# Seda (صدا): build and install the desktop app plus its Python sidecar.
#
#   make                 build app/target/release/seda (as you, never as root)
#   make install         build, then install under $(PREFIX) (asks for sudo)
#   sudo make install    same; the build still runs as $SUDO_USER
#   make install DESTDIR=/tmp/stage   staged install, no sudo
#   make uninstall       remove everything install put down

PREFIX      ?= /usr/local
DESTDIR     ?=
BINDIR      ?= $(PREFIX)/bin
DATADIR     ?= $(PREFIX)/share
SIDECAR_DIR ?= $(DATADIR)/seda/sidecar

APPS_DIR    := $(DATADIR)/applications
ICONS_DIR   := $(DATADIR)/icons/hicolor
ICON_SIZES  := 16 24 32 48 64 128 256 512

BINARY      := app/target/release/seda
ICON_SRC    := app/assets/icons/app

# No sudo when already root or staging into DESTDIR.
SUDO := $(if $(DESTDIR),,$(if $(filter 0,$(shell id -u)),,sudo))

# sudo resets the environment (MAKEFLAGS too), so hand the paths over explicitly.
SUBMAKE = $(MAKE) --no-print-directory PREFIX='$(PREFIX)' DESTDIR='$(DESTDIR)' \
	BINDIR='$(BINDIR)' DATADIR='$(DATADIR)' SIDECAR_DIR='$(SIDECAR_DIR)'

.PHONY: all build install install-files uninstall refresh-caches help

all: build

# The release build bakes SIDECAR_DIR in as the default sidecar project. Under
# sudo it runs as $SUDO_USER with their HOME and cargo on PATH, so app/target
# never ends up root-owned.
build:
	@set -e; \
	if [ "$$(id -u)" != 0 ]; then \
		cd app && SEDA_SIDECAR_DIR='$(SIDECAR_DIR)' cargo build --release; \
	elif [ -n "$$SUDO_USER" ] && [ "$$SUDO_USER" != root ]; then \
		home=$$(getent passwd "$$SUDO_USER" | cut -d: -f6); \
		echo "building as $$SUDO_USER"; \
		sudo -u "$$SUDO_USER" env HOME="$$home" \
			PATH="$$home/.cargo/bin:$$PATH" SEDA_SIDECAR_DIR='$(SIDECAR_DIR)' \
			sh -c 'cd app && cargo build --release'; \
	elif [ -x '$(BINARY)' ]; then \
		echo "warning: running as root without SUDO_USER; installing the existing $(BINARY) unrebuilt" >&2; \
	else \
		echo "error: refusing to build as root. Run 'make' as your user first, or use 'sudo make install' (which builds as \$$SUDO_USER)." >&2; \
		exit 1; \
	fi

install: build
	$(SUDO) $(SUBMAKE) install-files
	@$(if $(DESTDIR),,$(SUDO) $(SUBMAKE) refresh-caches)

# Internal: the copy step, run with $(SUDO) by install.
install-files:
	install -Dm755 $(BINARY) $(DESTDIR)$(BINDIR)/seda
	rm -rf $(DESTDIR)$(SIDECAR_DIR)/src
	install -Dm644 -t $(DESTDIR)$(SIDECAR_DIR) sidecar/pyproject.toml sidecar/uv.lock sidecar/README.md
	cd sidecar && find src \( -name .venv -o -name __pycache__ \) -prune -o \
		-type f ! -name '*.pyc' -exec install -Dm644 {} $(DESTDIR)$(SIDECAR_DIR)/{} \;
	install -Dm644 packaging/seda.desktop $(DESTDIR)$(APPS_DIR)/seda.desktop
	install -Dm644 $(ICON_SRC)/seda.svg $(DESTDIR)$(ICONS_DIR)/scalable/apps/seda.svg
	for n in $(ICON_SIZES); do \
		install -Dm644 $(ICON_SRC)/png/seda-$$n.png $(DESTDIR)$(ICONS_DIR)/$${n}x$${n}/apps/seda.png || exit 1; \
	done

uninstall:
	$(SUDO) rm -f $(DESTDIR)$(BINDIR)/seda
	$(SUDO) rm -rf $(DESTDIR)$(SIDECAR_DIR)
	-$(SUDO) rmdir $(DESTDIR)$(DATADIR)/seda 2>/dev/null
	$(SUDO) rm -f $(DESTDIR)$(APPS_DIR)/seda.desktop
	$(SUDO) rm -f $(DESTDIR)$(ICONS_DIR)/scalable/apps/seda.svg
	$(SUDO) rm -f $(foreach n,$(ICON_SIZES),$(DESTDIR)$(ICONS_DIR)/$(n)x$(n)/apps/seda.png)
	@$(if $(DESTDIR),,$(SUDO) $(SUBMAKE) refresh-caches)

# Best effort: missing tools or failures are ignored.
refresh-caches:
	-@command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q -t -f $(ICONS_DIR) || true
	-@command -v update-desktop-database >/dev/null && update-desktop-database -q $(APPS_DIR) || true

help:
	@echo "Targets:"
	@echo "  build       cargo build --release in app/ (default; runs as \$$SUDO_USER under sudo)"
	@echo "  install     build, then install binary, sidecar, desktop file and icons"
	@echo "  uninstall   remove everything install put down"
	@echo "  help        this text"
	@echo "Variables (current value):"
	@echo "  PREFIX      $(PREFIX)"
	@echo "  DESTDIR     $(DESTDIR)   (staging root; set it to install without sudo)"
	@echo "  BINDIR      $(BINDIR)"
	@echo "  DATADIR     $(DATADIR)"
	@echo "  SIDECAR_DIR $(SIDECAR_DIR)   (baked into the binary at build time)"
