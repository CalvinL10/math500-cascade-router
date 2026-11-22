# MATH 550 Final Presentation Slides (HTML/CSS → Playwright PDF)

## Deliverables

- **[math550_presentation.pdf](math550_presentation.pdf)** — 15-page 16:9 widescreen PDF (`960 × 540 pt`, **Slides 1–10** main presentation + **Backup B1–B5** for Q&A, with 40 clickable internal PDF navigation links).
- **[speaker_cue_cards.pdf](speaker_cue_cards.pdf)** & **[speaker_cue_cards.md](speaker_cue_cards.md)** — 3-page printable/mobile-friendly podium cue cards (`~7:30` zero-decimal elementary English speaking pace for the 15-min class slot, Chinese pointing & logic anchors, easy Q&A defense lines, emergency stage rescue lines, and the 7 safety rules).
- **[deck.html](deck.html)** & **[styles.css](styles.css)** — Source 16:9 HTML/CSS deck with locally bundled KaTeX (`0.19.1`) and self-hosted `Inter` / `IBM Plex Mono` fonts (renders offline).

## Rebuilding After Edits

```powershell
python slides/build_assets.py
python slides/build_pdf.py
```

`build_pdf.py` runs an automated DOM overflow check, KaTeX verification, console/resource check, and internal PDF link verification before saving both PDFs and 180-DPI previews under `preview_pdf/`.
