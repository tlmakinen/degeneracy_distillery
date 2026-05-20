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
<!-- Simple edition (natural ratio ≈ 920×600 ⇒ ~3/2) -->
<iframe
  src="https://tlmakinen.github.io/degeneracy_distillery/illustration/index_simple.html"
  title="Hearing Degeneracy"
  loading="lazy"
  allow="autoplay"
  style="width: 100%; aspect-ratio: 3 / 2; min-height: 460px; border: 0; border-radius: 8px;"
></iframe>

<!-- Advanced edition (natural ratio ≈ 1100×860 ⇒ ~5/4) -->
<iframe
  src="https://tlmakinen.github.io/degeneracy_distillery/illustration/index_tape.html"
  title="Hearing Degeneracy (advanced)"
  loading="lazy"
  allow="autoplay"
  style="width: 100%; aspect-ratio: 5 / 4; min-height: 700px; border: 0; border-radius: 8px;"
></iframe>
```

The pages auto-detect when they're loaded inside an `<iframe>`
(`window.self !== window.top`) and switch into a compact "embed mode" that:

- skips the simple/advanced chooser overlay (the host already chose),
- hides the lede paragraph,
- **proportionally scales the entire rig down via CSS `zoom`** so the natural
  desktop layout (decks side-by-side, mixer in the middle) shrinks to
  whatever width the blog gives the iframe — JS recomputes the scale on
  every resize / ResizeObserver tick,
- exposes an **↗ Open in new tab** button in the header that opens the
  standalone, full-size version of the demo.

Design widths (each edition is laid out at this width and zoomed down):

- `index_tape.html` — 1100 px
- `index_simple.html` — 920 px

Tips:

- **Pick the entry point per post** (link directly to `index_simple.html` or
  `index_tape.html`); the chooser overlay only really helps when the demo is
  visited standalone.
- **`loading="lazy"`** keeps math.js + fonts off the critical path.
- **Audio policy**: the demo requires a user click on `POWER ON` before any
  audio is created, so browser autoplay restrictions are satisfied without
  extra work. `allow="autoplay"` is belt-and-suspenders.
- **Sizing**: because of the proportional scaling, the iframe height should
  track its width. Use `aspect-ratio` (3/2 for simple, 5/4 for advanced)
  with a `min-height` floor for very narrow viewports. The popout button is
  always available if a reader wants the un-scaled experience.
- **Browser support**: scaling uses CSS `zoom`, which is supported in
  Chrome, Edge, Safari, and Firefox 126+ (May 2024). Older browsers fall
  back to overflow-clipping and will only show the left portion of the
  rig — push readers to the popout button.

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
