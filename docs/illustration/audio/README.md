# Audio clips for the tape defaults

Drop short MP3 / OGG / WAV files in this folder and reference them from
[`../tape_defaults.html`](../tape_defaults.html). Same-origin paths only —
GitHub Pages serves these files with CORS-friendly headers automatically,
so the demo can route them through the Web Audio API fader chain.

## Recommendations

- **Length**: 30–60 seconds works well. The demo loops the deck, so pick a
  clip that loops cleanly (matched musical bar, no hard cut).
- **Bitrate**: 128 kbps MP3 keeps the page light without obvious artefacts.
- **Mono vs stereo**: either is fine; the demo just routes the channel into
  the deck's gain node.
- **Filename**: anything works, but the example config expects
  `track1.mp3` … `track4.mp3`. Edit `tape_defaults.html` if you want
  different names.

## What does *not* go here

Don't drop full-length copyrighted music here unless you have rights to
distribute it. Use Spotify links (via the `spotifyUrl` field in the config)
to point to the full song on Spotify instead.
