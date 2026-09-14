# LightGlue Assets

This folder contains all LightGlue-specific project assets.

## Structure

- `docs/`: LightGlue design, safety, and implementation notes.
- `scripts/`: LightGlue utility scripts:
  - `create_marker_templates.py`
  - `test_lightglue_fallback.py`
  - `validate_lightglue_on_session.py`
- `lightglue_bundle.tar.gz`: Compressed bundle of LightGlue docs, scripts, example config, and runtime fallback module.

## Typical commands

Generate templates:

```bash
python lightglue/scripts/create_marker_templates.py --marker-ids 0 1 2 3 --dict 4x4_50 --output-dir templates/markers --size 400
```

Validate on saved session:

```bash
python lightglue/scripts/validate_lightglue_on_session.py --session data/sessions/<session_name> --templates templates/markers
```
