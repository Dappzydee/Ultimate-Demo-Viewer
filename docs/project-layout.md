# Project layout

This repository separates supported software from experiments and local data.

```text
cs2_visibility/  Reusable, supported Python library and command-line modules.
viewer/           Dependency-free browser viewer assets.
scripts/          Maintainer-only diagnostics, utilities, and archived prototypes.
tests/            Automated tests; keep fixtures small and redistributable.
docs/             Architecture and developer documentation.
examples/         Non-authoritative scratch commands and sample workflows.
out/              Generated artifacts; ignored by Git.
data/             Optional local demos and large inputs; ignored by Git.
```

Add a feature to `cs2_visibility/` first. Keep a small root wrapper only when it is a supported checkout command, and optionally give it a package-backed command-line entry point in `pyproject.toml`. Put throwaway investigation code in `scripts/development/`, and move superseded experiments to `scripts/legacy/` rather than adding more files at the repository root.

The `viewer/` folder remains separate because it is a small standalone web application. Keep its browser assets there unless it is intentionally turned into a separately packaged frontend.
