Concmath-otf package
====================

## Description

`Concrete-Math.otf’ is an Opentype version of the Concrete Math font
created by Ulrik Vieth in MetaFont. `concmath-otf.sty’ is a replacement
for the original `concmath.sty’ package.

## Contents

* Concrete-Math.otf    OpenType Math font
* concmath-otf.sty     LaTeX style file: replaces concmath.sty for LuaTeX/XeTeX
* concmath-otf.pdf     Documentation in PDF format
* concmath-otf.ltx     LaTeX source of concmath-otf.pdf
* unimath-concrete.pdf Modified version of unimath-symbols.pdf
                       showing available Concrete-Math symbols compared to
		               LatinModern, STIXTwo, TeXGyrePagella and DejaVu.
* unimath-concmath.ltx LaTeX source of unimath-concrete.pdf
* README.md            (this file)

## Installation

This package is meant to be installed automatically by TeXLive, MikTeX, etc.
Otherwise, the package can be installed under TEXMFHOME or TEXMFLOCAL, f.i.
Concrete-Math.otf in directory  texmf-local/fonts/opentype/public/concmath-otf/
and concmath-otf.sty in directory  texmf-local/tex/latex/concmath-otf/.  
Documentation files and their sources can go to directory
texmf-local/doc/fonts/public/concmath-otf/

Don't forget to rebuild the file database (mktexlsr or so) if you install
under TEXMFLOCAL.

Finally, make the system font database aware of the Concrete-Math font
(fontconfig under Linux).

## License

* The font `Concrete-Math.otf’ is licensed under the SIL Open Font License,
Version 1.1. This license is available with a FAQ at:
http://scripts.sil.org/OFL
* The other files are distributed under the terms of the LaTeX Project
Public License from CTAN archives in directory macros/latex/base/lppl.txt.
Either version 1.3c or, at your option, any later version.

## Changes

* First public version: 0.20

* v. 0.21
  - Integrals are now slanted by default, option "Style=upint" (+ss03)
    make them upright.
  - More integrals added: U+222F to U+2233 and U+2A11.
  - Corrected symbols prime, dprime, etc. (U+2032 to U+2037).
* v. 0.22
  - concmath-otf.sty loads Concrete-Math.otf by file name for XeTeX.
  - Glyphs corrections: \sum, \prod, \coprod and \amalg, \infty,
    \propto, \wp, \ell redesigned;
* v. 0.23
  - Delimiters, integrals, sum, prod etc. are now vertically
    centred by design on the maths axis (required by luametatex).
  - Upright integrals: fixed left bearings.
* v. 0.24
  - Fixed \wideoverbar (U+0305) which failed with XeTeX.
  - Corrected negative right bearings of some display integrals.
  - Added stretchable integral for U+222B (usable with luametatex).
* v. 0.25
  - Package now compatible with mathtools’ stretchable arrows.
  - Corrected some glyphs’ inconsistencies (maths axis, etc.).
  - \diagup and \diagdown moved from private area to U+27CB, U+27CD.
* v. 0.40 
  - Added missing stretchable glyhs: \wideoverbar, \widecheck, \widebreve, 
    \overrightharpoon, \overleftharpoon, \overleftrightharpoon, 
    \underrightharpoon, \underleftharpoon, \underleftrightharpoon, 
    \underrightarrow, \underleftarrow, \underleftrightarrow, 
    \mathunderbar, \underbracket, \wideutilde.
  - \overbrace, \underbrace redrawn (all sizes).
  - Metrics corrected:  \overparen, \overrightarrow, \overleftarrow,
    \overleftrightarrow.
  - Many glyphs have been emboldened (suggested by Mikael Sundqvist and
    Hans Hagen).
  - Most horizontal arrows and harpoons are now stretchable (dito).
  - Glyphs \leqqslant, \geqqslant, \nleqqslant, \ngeqqslant added.
* v0.41:
  - radicals corrected.
  - concmath-otf.sty: options handled by `l3keys` instead of `xkeyval` 
    (requires a LaTeX kernel not older than 2022-06-01).
* v0.50:
  - Digits and Lowercase Blackboard Bold glyphs added (adapted from 
    Erewhon-Math).
  - Many math symbols added, the coverage now compares to the TeXGyre fonts.
  - Sans-serif alphabets (Latin and Greek) added, they are derived from
    Iwona text fonts.
  - Typewriter alphabet (Latin) added, derived from RobotoMono-Regular.
  - Bold Script (Uppercase only) glyphs added.
* v0.61:
  - Fixed IsExtendedShape flags.
  - \mid (U+2223) and \parallel (U+2225) vertical variants changed.
  - Blackboard bold capitals redesigned from scratch to match Concrete 
	upright capitals.
  - Bold glyphs redesigned.
  - Concrete-Math-Bold.otf added (minimal subset of glyphs).
  - Superscripts get a suitable boldness (ssty feature enhanced).
  - Inconsistencies between vertical delimiter sizes fixed.
  - Metrics of \widebreve, \widecheck, \widehat, \widetilde first
    horizontal variant (.h0) changed: f.i. $\hat{r} \ne \widehat{r}$.
* v0.62:
  - Further corrections of vertical delimiter sizes.
* v0.63:
  - Fixed Game glyph (U+2141).
  - \leqslant, \geqslant (and related glyphs) are now vertically aligned 
    with \leq, \geq.
  - Glyph mfrakZ (U+2128) corrected.
  - Concrete-Math-Bold: missing italic correction added for int.up.
  - Concrete-Math-Bold: added missing features +ss03, +ss04, +ss05.
* v0.64:
  - Added lmoustache, rmoustache (U+23B0, U+23B1).
  - Added \lgroup (U+27EE), \rgroup (U+27EF), and their vertical extensions.
  - Overbraces and underbraces resized.
  - Fixed inconsistency in composed radicals.
* v0.65:
  - Fixed notaccent (U+0338).
  - bar (U+0304), wideoverbar (U+0305), mathunderbar (U+0332) thicknesses' 
    unified to default rule thickness.
  - Glyphs U+2032 to U+2037 and U+2057 (prime and co.) resized so that f.i. 
    \(f'\), \(f\prime\) and \(f^{\prime}\) produce the same output.
* v0.66:
  - Up and down arrows variants enhanced.
  - Math table fixes for horizontal composition of over/underbrace.
* v0.70:
  - Greek msans and mitsans glyphs added in Private Area.
  - Invisible Math operators (U-2061 to U+2064) added.
  - Compatibility with Typst v0.14: all glyphs used in
    vertical compositions now sit on the baseline.
  - Documentation fixes.
* v0.73:
  - added missing glyph U+200B (nulspace).
  - Math table: TopRightVertex and BottomRightVertex tuned for italic glyphs.
  - Inconsistencies in widehat, widetilde and other over/under accents fived.
  
---
Copyright 2022-2026  Daniel Flipo  
E-mail: daniel (dot) flipo (at) free (dot) fr
