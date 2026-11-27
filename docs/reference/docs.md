# Building these docs

Pages mix three sources:

- **Docstrings** (Google style) become the [API reference](api.md) through mkdocstrings.
- **Scripts in `docs/scripts/`** are complete, runnable analyses. Pages include sections of them with snippets:

    ```text
    --8<-- "docs/scripts/rocking_curve.py:fit"
    ```

    and the script marks the section with `# --8<-- [start:fit]` … `# --8<-- [end:fit]`. Each script also writes its
    figures and printed results to `docs/assets/generated/`, which pages include, and ends with `assert`s.
- **Prose** in `docs/**/*.md`.

`tests/test_docs_scripts.py` runs every script, so a page cannot show code that no longer works. A missing snippet or
generated file fails the build (`check_paths = true`).

```sh
uv sync --extra docs
uv run python tools/build_docs.py         # run the scripts, then zensical build into site/
uv run python tools/build_docs.py serve   # live preview
```

To add a page: write `docs/scripts/<topic>.py` with section markers, outputs written via `_common.save` and
`_common.text`, and checks at the end. Then write the page, and add it to `nav` in `zensical.toml`.
