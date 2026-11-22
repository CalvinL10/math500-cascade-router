"""Compile deck.html to 16:9 math550_presentation.pdf via Playwright Chromium,
verify zero overflow across every slide, inspect internal PDF links, and export
180-DPI page previews rendered directly from the generated PDF."""

from __future__ import annotations

from pathlib import Path
from playwright.sync_api import sync_playwright
import pymupdf

SLIDES_DIR = Path(__file__).resolve().parent
DECK_HTML = SLIDES_DIR / "deck.html"
OUT_PDF = SLIDES_DIR / "math550_presentation.pdf"
CUE_HTML = SLIDES_DIR / "speaker_cue_cards.html"
CUE_PDF = SLIDES_DIR / "speaker_cue_cards.pdf"
PREVIEW_DIR = SLIDES_DIR / "preview_pdf"


def main() -> None:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for old_png in PREVIEW_DIR.glob("*.png"):
        old_png.unlink()

    url = DECK_HTML.as_uri()
    console_errors: list[str] = []
    page_errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.goto(url, wait_until="networkidle")
        page.wait_for_function("() => window.__katexReady === true")
        page.evaluate("() => document.fonts.ready")
        page.wait_for_timeout(250)

        # Verify every slide has zero overflow, KaTeX rendered, no raw $, and all images loaded
        audit = page.evaluate(
            """() => {
                const slides = Array.from(document.querySelectorAll('.slide'));
                const issues = [];
                slides.forEach((s, idx) => {
                    const id = s.id || `slide-${idx+1}`;
                    const sh = s.scrollHeight, ch = s.clientHeight;
                    const sw = s.scrollWidth, cw = s.clientWidth;
                    if (sh > ch + 1 || sw > cw + 1) {
                        issues.push(`${id}: overflow scroll=(${sw}x${sh}) client=(${cw}x${ch})`);
                    }
                    const footer = s.querySelector('.slide-footer');
                    const body = s.querySelector('.slide-body');
                    if (footer && body) {
                        const fr = footer.getBoundingClientRect();
                        const br = body.getBoundingClientRect();
                        if (br.bottom > fr.top - 4) {
                            issues.push(`${id}: body bottom (${br.bottom.toFixed(1)}) encroaches footer top (${fr.top.toFixed(1)})`);
                        }
                    }
                    // Check for unrendered LaTeX delimiters outside KaTeX nodes
                    const walker = document.createTreeWalker(s, NodeFilter.SHOW_TEXT);
                    let node;
                    while ((node = walker.nextNode())) {
                        const parent = node.parentElement;
                        if (parent && (parent.closest('.katex') || parent.tagName === 'SCRIPT' || parent.tagName === 'STYLE')) continue;
                        if (node.nodeValue && node.nodeValue.includes('$')) {
                            issues.push(`${id}: unrendered '$' in text: ${node.nodeValue.trim().slice(0, 80)}`);
                        }
                    }
                });
                const imgs = Array.from(document.querySelectorAll('img'));
                imgs.forEach(img => {
                    if (!img.complete || img.naturalWidth === 0) {
                        issues.push(`Image failed to load: ${img.src}`);
                    }
                });
                const katexCount = document.querySelectorAll('.katex').length;
                if (katexCount === 0) {
                    issues.push('No KaTeX nodes rendered!');
                }
                const katexErrs = document.querySelectorAll('.katex-error');
                if (katexErrs.length > 0) {
                    issues.push(`Found ${katexErrs.length} KaTeX render errors`);
                }
                return { totalSlides: slides.length, totalImgs: imgs.length, katexCount, issues };
            }"""
        )
        if page_errors or console_errors:
            audit["issues"].extend([f"pageerror: {e}" for e in page_errors])
            audit["issues"].extend([f"console.error: {e}" for e in console_errors])
        print("DOM Audit:", audit)

        page.pdf(
            path=str(OUT_PDF),
            prefer_css_page_size=True,
            print_background=True,
        )

        # Render Speaker Cue Cards PDF
        cue_page = browser.new_page(viewport={"width": 900, "height": 1100})
        cue_errors: list[str] = []
        cue_page.on("pageerror", lambda exc: cue_errors.append(str(exc)))
        cue_page.on("console", lambda msg: cue_errors.append(msg.text) if msg.type == "error" else None)
        cue_page.goto(CUE_HTML.as_uri(), wait_until="networkidle")
        cue_page.wait_for_function("() => window.__katexReady === true")
        cue_page.evaluate("() => document.fonts.ready")
        cue_page.wait_for_timeout(200)
        cue_unrendered = cue_page.evaluate(
            """() => {
                const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
                const bad = [];
                let node;
                while ((node = walker.nextNode())) {
                    const parent = node.parentElement;
                    if (parent && (parent.closest('.katex') || parent.tagName === 'SCRIPT' || parent.tagName === 'STYLE')) continue;
                    if (node.nodeValue && node.nodeValue.includes('$')) bad.push(node.nodeValue.trim().slice(0, 60));
                }
                return { katex: document.querySelectorAll('.katex').length, bad };
            }"""
        )
        print("Cue Cards Audit:", cue_unrendered, "errors:", cue_errors)
        if cue_errors or cue_unrendered["bad"]:
            audit["issues"].extend(cue_errors)
            audit["issues"].extend([f"Cue unrendered $: {x}" for x in cue_unrendered["bad"]])
        cue_page.pdf(
            path=str(CUE_PDF),
            prefer_css_page_size=True,
            print_background=True,
        )
        browser.close()

    cue_doc = pymupdf.open(str(CUE_PDF))
    print(f"Cue PDF generated: {CUE_PDF} | pages={len(cue_doc)}")
    for i, cpage in enumerate(cue_doc):
        pix = cpage.get_pixmap(dpi=150, alpha=False)
        pix.save(str(PREVIEW_DIR / f"cue_{i + 1:02d}.png"))

    doc = pymupdf.open(str(OUT_PDF))
    print(f"PDF generated: {OUT_PDF} | pages={len(doc)}")
    total_links = 0
    bad_links: list[str] = []
    for i, pdf_page in enumerate(doc):
        links = pdf_page.get_links()
        total_links += len(links)
        for lk in links:
            target_page = lk.get("page")
            if target_page is None or not (0 <= target_page < len(doc)):
                bad_links.append(f"Page {i+1}: invalid link {lk}")
        pix = pdf_page.get_pixmap(dpi=180, alpha=False)
        pix.save(str(PREVIEW_DIR / f"page_{i + 1:02d}.png"))
        dests = [f"{lk.get('nameddest')}->p{lk.get('page', -1)+1}" for lk in links]
        print(f"  Page {i + 1:02d}: rect={tuple(round(x, 1) for x in pdf_page.rect)}, links={dests}")
    print(f"Total internal PDF links: {total_links}")
    if bad_links:
        audit["issues"].extend(bad_links)
    if audit["issues"]:
        raise RuntimeError("Layout/render issues detected:\n" + "\n".join(audit["issues"]))


if __name__ == "__main__":
    main()
