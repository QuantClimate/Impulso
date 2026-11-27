"""Sphinx configuration for the Impulso documentation.

Single-engine build: MyST-NB parses and executes the tutorials, Sphinx renders
everything. No Quarto, no pandoc, no markdown post-processing.
"""

from __future__ import annotations

import os
import re

# -- Project information -----------------------------------------------------
project = "impulso"
author = "Thomas Pinder"
copyright = "2026, Thomas Pinder"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_nb",  # MyST markdown + executable notebooks (enables myst_parser)
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # Google-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_codeautolink",  # link API names in example code to the reference
    "sphinx_sitemap",  # emit sitemap.xml
    "sphinxext.opengraph",  # og: link-preview metadata
]

# -- Bibliography (sphinxcontrib-bibtex) -------------------------------------
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"  # {cite:t} -> Uhlig (2005)
bibtex_default_style = "unsrt"

templates_path = ["_templates"]

# Tutorials are jupytext py:percent notebooks; MyST-NB reads them via jupytext.
source_suffix = {
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
    ".py": "myst-nb",
    ".rst": "restructuredtext",  # autosummary-generated API stubs
}
nb_custom_formats = {".py": ["jupytext.reads", {"fmt": "py:percent"}]}

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "conf.py",  # this config module is not a document
    "**.ipynb_checkpoints",
    "agents/**",  # internal agent docs, not part of the site
    "adr/**",  # architecture decision records live in-repo, not on the site
    # Underscore-prefixed .py in tutorials are helper modules imported by a
    # notebook at runtime, not notebooks themselves — keep MyST-NB from
    # collecting them as standalone documents (they'd warn as orphan pages).
    "tutorials/_*.py",
]

# -- MyST / MyST-NB ----------------------------------------------------------
myst_enable_extensions = [
    "dollarmath",  # $...$ and $$...$$ — the feature Quarto+Zensical broke
    "amsmath",  # \begin{align} etc.
    "colon_fence",  # ::: {note} admonitions
    "deflist",
    "tasklist",
    "html_image",
    "attrs_inline",
    "substitution",
]
myst_dmath_double_inline = True
myst_heading_anchors = 3  # auto-slug headings so in-page [](#anchor) links resolve

# Execute notebooks and cache results in a gitignored cache. Source .md carry
# no outputs; a cell re-runs only when its code changes.
nb_execution_mode = "cache"
# Smoke (IMPULSO_DOCS_CI=1) and full renders execute the SAME notebook source
# but with very different MCMC — yet jupyter-cache keys on source alone, so a
# shared cache lets a tiny smoke run (draws=10) poison the full-fidelity build.
# Separate the cache directories so the two modes can never overwrite one
# another (locally or in CI).
_smoke_render = os.environ.get("IMPULSO_DOCS_CI") == "1"
nb_execution_cache_path = os.path.join(
    os.path.dirname(__file__),
    "_build",
    ".jupyter_cache_ci" if _smoke_render else ".jupyter_cache",
)
nb_execution_timeout = 1800  # heavy MCMC tutorials
# Strict by default (PR gate + local): a failed cell fails the build. On the
# deploy path (IMPULSO_DOCS_RESILIENT=1) we do NOT raise, so one slow/broken
# MCMC notebook cannot block the whole site — it renders an error cell while
# every other page (and the last-good cached output) still deploys.
nb_execution_raise_on_error = os.environ.get("IMPULSO_DOCS_RESILIENT") != "1"
nb_merge_streams = True

# -- Autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Numbered figures, tables and equations ----------------------------------
numfig = True
math_eqref_format = "Eq. {number}"

# -- sphinx-codeautolink -----------------------------------------------------
# Adds a "Examples using …" backreference block to each documented object.
codeautolink_autodoc_inject = True

# -- Intersphinx -------------------------------------------------------------
# Each mapping lists two inventory locations: the upstream one (`None` means
# "<target_uri>objects.inv"), then a vendored copy under `_inventories/`.
# intersphinx tries them in order and stops at the first that loads.
#
# The fallback exists because the PR build runs under `-W`, where an unreachable
# inventory is a hard failure: a docs.python.org outage (503 on every URL) once
# failed build-docs on every open PR. The warning carries no type, so
# `suppress_warnings` cannot target it. If some locations fail but one loads,
# intersphinx logs only info, so the vendored copy keeps the build green.
# GPJax does the same (its docs/conf.py has the details).
#
# Cost: the vendored copies go stale, so an object added upstream does not
# resolve on a build where the fetch failed. Refresh them by re-downloading each
# `<target_uri>objects.inv` into docs/_inventories/.
_INV = os.path.join(os.path.dirname(__file__), "_inventories")
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", (None, f"{_INV}/python.inv")),
    "numpy": ("https://numpy.org/doc/stable/", (None, f"{_INV}/numpy.inv")),
    "pandas": ("https://pandas.pydata.org/docs/", (None, f"{_INV}/pandas.inv")),
    "arviz": ("https://python.arviz.org/en/stable/", (None, f"{_INV}/arviz.inv")),
    "pymc": ("https://www.pymc.io/projects/docs/en/stable/", (None, f"{_INV}/pymc.inv")),
    "matplotlib": ("https://matplotlib.org/stable/", (None, f"{_INV}/matplotlib.inv")),
}

# -- Link check --------------------------------------------------------------
# www.stlouisfed.org did not answer the `docs-linkcheck` job's requests from
# GitHub's runners within linkcheck's 30-second timeout on two consecutive runs,
# so the job skips that host.
linkcheck_ignore = [r"https://www\.stlouisfed\.org/"]

# -- HTML output -------------------------------------------------------------
html_theme = "shibuya"
html_title = "impulso"
# Sitemap, canonical link and og:url (via ogp_site_url) all derive from this, so
# it must match the custom domain set in the repo's Pages settings. The og:image
# in tutorials/proxy-svar.py is written out in full and must match it too.
html_baseurl = "https://impulso.quantclimate.com/"
sitemap_url_scheme = "{link}"

# -- Open Graph link previews (sphinxext-opengraph) ---------------------------
# The extension emits og:title/og:url/og:description for every page. A page
# opts into a preview image with a docutils `meta` directive in its first
# markdown cell. MyST-NB parses neither notebook front matter nor a mocked
# `{meta}` directive for the jupytext tutorials, so the directive goes
# through `{eval-rst}` (the escape hatch myst-parser itself recommends):
#     ```{eval-rst}
#     .. meta::
#        :property=og:image: https://impulso.quantclimate.com/_static/<name>.png
#     ```
# The image itself is written by the notebook into `stylesheets/` (the
# html_static_path dir) and committed, so the URL stays valid even when a
# warm jupyter-cache skips the cell that regenerates it.
ogp_site_url = html_baseurl
ogp_enable_meta_description = True
# Off until we choose a template: with matplotlib installed the extension
# would otherwise auto-generate a text card for every page.
ogp_social_cards = {"enable": False}
html_static_path = ["stylesheets"]
html_css_files = ["extra.css"]
# The theme links the SVG favicon from every page. The ICO and the iOS
# home-screen icon go to the site root, where browsers look for them by
# default. Both are rendered from the SVG; after editing it, regenerate them:
#     cd docs/_brand
#     for s in 16 32 48; do rsvg-convert -w $s -h $s favicon.svg -o /tmp/f$s.png; done
#     magick /tmp/f16.png /tmp/f32.png /tmp/f48.png favicon.ico
#     sed 's/ rx="7"//g' favicon.svg | rsvg-convert -w 180 -h 180 -o apple-touch-icon.png
# (The apple-touch-icon is square-cornered because iOS applies its own mask.)
html_favicon = "_brand/favicon.svg"
# Cloudflare serves the site; _redirects maps directory URLs to index.html and
# _headers adds the CORS header that GitHub Pages used to send.
html_extra_path = ["_redirects", "_headers", "_brand/favicon.ico", "_brand/apple-touch-icon.png"]
html_theme_options = {
    "accent_color": "crimson",  # names the token family; stylesheets/extra.css re-tones the crimson scale to ledger oxblood
    "color_mode": "auto",  # follow the reader's light/dark preference
    "github_url": "https://github.com/QuantClimate/Impulso",
    "nav_links": [
        {"title": "PyPI", "url": "https://pypi.org/project/impulso"},
    ],
}
html_context = {
    "github_user": "QuantClimate",
    "github_repo": "Impulso",
    "github_version": "main",
    "doc_path": "docs",
}

# Disable the sampler progress widget in all rendered notebooks (inherited by
# the myst-nb execution kernel).
os.environ["IMPULSO_DOCS_BUILD"] = "1"

# Smoke-render flag for CI (mirrors the old Quarto `ci` parameter). Tutorials
# read this to shrink MCMC when set.
os.environ.setdefault("IMPULSO_DOCS_CI", "0")

# -- Render-mode stamp --------------------------------------------------------
# Belt-and-braces against smoke renders masquerading as the production docs:
# smoke builds carry a visible banner, and every HTML build writes
# `render-mode.txt` at the site root so what is actually deployed can be
# checked with `curl <site>/render-mode.txt`.
if _smoke_render:
    html_theme_options["announcement"] = (
        "Smoke render: MCMC is shrunk for CI speed, so figures and diagnostics "
        "are not publication-fidelity. The production docs are built with full sampling."
    )


# -- Lazy-loaded figure images ------------------------------------------------
# Tutorial figures are 0.5-1.4MB PNGs, so let the browser defer them until
# they scroll into view. Only `<img>` tags whose src points into `_images/`
# (Sphinx's directory for content figures) are touched — theme/branding
# images in the page chrome are left alone. Skipping tags that already carry
# a `loading` attribute keeps the rewrite idempotent.
_IMG_TAG_RE = re.compile(r"<img\b[^>]*>")


def _lazy_load_figures(app, pagename, templatename, context, doctree):
    """Add `loading="lazy"` to content-figure img tags in a page's body.

    Args:
        app: The Sphinx application.
        pagename: Name of the page being rendered.
        templatename: Name of the template rendering the page.
        context: Jinja context; `context["body"]` holds the page HTML.
        doctree: Doctree of the page, or None for pages without a source.
    """
    body = context.get("body")
    if not body:
        return

    def _annotate(match):
        tag = match.group(0)
        if "_images/" not in tag or "loading=" in tag:
            return tag
        return tag.replace("<img", '<img loading="lazy"', 1)

    context["body"] = _IMG_TAG_RE.sub(_annotate, body)


def setup(app):
    """Register the render-mode marker and lazy-image hooks."""

    def _write_render_mode(app, exception):
        if exception is None and app.builder.name == "html":
            mode = "smoke" if _smoke_render else "full"
            with open(os.path.join(app.outdir, "render-mode.txt"), "w") as fh:
                fh.write(mode + "\n")

    app.connect("build-finished", _write_render_mode)
    # html-page-context only fires for HTML builders, so this is inert for
    # linkcheck and any other non-HTML build.
    app.connect("html-page-context", _lazy_load_figures)
