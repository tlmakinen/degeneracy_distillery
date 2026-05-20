# Hearing Degeneracy — Interactive Demo

Static, single-file HTML demos that illustrate **degeneracy** (different parameter
combinations producing identical observables) by routing audio through a fader
interaction. Designed to be embedded in a blog post.

## Entry points

| File | Audience | What it does |
| --- | --- | --- |
| [`index_simple.html`](./index_simple.html) | Non-technical readers | Two channels, four preset "scenarios" (linear, collapse, sine, crowded) — no math required. |
| [`index_tape.html`](./index_tape.html) | Default landing page | Tape-deck UI with a chooser that routes to either edition. Use this as the standalone link. |
| [`index.html`](./index.html) | Technical / blog body | Synthwave/CRT styling. User-editable equation `f(f1, f2, f3)`, three faders, waveform + degeneracy heatmaps. |

Both editions support default 220 Hz / 330 Hz synth tones (no upload needed) and
optional user-uploaded audio per channel.

## Hosting on GitHub Pages

The repo's Pages source is the `docs/` folder, so once Pages is enabled the
files publish at:

- https://tlmakinen.github.io/degeneracy_distillery/illustration/index_tape.html
- https://tlmakinen.github.io/degeneracy_distillery/illustration/index_simple.html
- https://tlmakinen.github.io/degeneracy_distillery/illustration/index.html

To enable: **Repo → Settings → Pages → Source: `Deploy from a branch`,
Branch: `main`, Folder: `/docs`**.

## Embedding in a blog post

Markdown supports raw HTML, so paste an `<iframe>` straight into your post:

```html
<!-- Simple edition: tighter, fits ~640 px tall -->
<iframe
  src="https://tlmakinen.github.io/degeneracy_distillery/illustration/index_simple.html"
  title="Hearing Degeneracy"
  loading="lazy"
  allow="autoplay"
  style="width: 100%; min-height: 640px; border: 0; border-radius: 8px;"
></iframe>

<!-- Advanced edition: needs more vertical space for the mixer + heatmaps -->
<iframe
  src="https://tlmakinen.github.io/degeneracy_distillery/illustration/index_tape.html"
  title="Hearing Degeneracy (advanced)"
  loading="lazy"
  allow="autoplay"
  style="width: 100%; min-height: 1100px; border: 0; border-radius: 8px;"
></iframe>
```

The pages auto-detect when they're loaded inside an `<iframe>`
(`window.self !== window.top`) and switch into a compact "embed mode" that:

- skips the simple/advanced chooser overlay (the host already chose),
- hides the lede paragraph and tightens paddings,
- keeps the two tape decks **side-by-side** and wraps the mixer below them
  at typical blog content widths,
- exposes an **↗ Open in new tab** button in the header that opens the
  standalone, full-size version of the demo.

Tips:

- **Pick the entry point per post** (link directly to `index_simple.html` or
  `index_tape.html`); the chooser overlay only really helps when the demo is
  visited standalone.
- **`loading="lazy"`** keeps math.js + fonts off the critical path.
- **Audio policy**: the demo requires a user click on `POWER ON` before any
  audio is created, so browser autoplay restrictions are satisfied without
  extra work. `allow="autoplay"` is belt-and-suspenders.
- **Sizing**: use `min-height` rather than `aspect-ratio` — embed mode
  reflows mostly vertically once the mixer/master drops below the decks.
  ~640 px is enough for the simple edition; the advanced edition needs
  ~1100 px because of the equation panel and the heatmaps.

## Running locally

Pure static files — no build step. From the repo root:

```bash
python -m http.server 8000 -d docs
# open http://localhost:8000/illustration/index_tape.html
```

External dependencies are loaded from CDNs:

- [Google Fonts](https://fonts.google.com) — `VT323`, `Silkscreen`, `Press Start 2P`
- [`mathjs@13.2.2`](https://mathjs.org) — used by `index.html` and `index_tape.html`
  to safely parse user-supplied equations (the simple edition has no math.js
  dependency).

## Files

```
docs/illustration/
├── README.md          # this file
├── index.html         # synthwave/CRT edition (advanced, original styling)
├── index_simple.html  # tape-deck simple edition (presets, grandma-friendly)
└── index_tape.html    # tape-deck advanced edition + simple/advanced chooser
```
