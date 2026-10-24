# Vendored crates

## gpui-pre 0.3.9 (`vendor/gpui-pre`)

A copy of `gpui-pre` 0.3.9 from crates.io (the exact GPUI snapshot gpui-kit 0.8.0
pins), wired in with `[patch.crates-io]` in `app/Cargo.toml`. It adds right-to-left
layout, which upstream GPUI does not have. Every change is marked `[seda patch]`
(`grep -rn "seda patch" vendor/gpui-pre/src`).

Regenerate the diff with:

```sh
diff -ru ~/.cargo/registry/src/index.crates.io-*/gpui-pre-0.3.9/src vendor/gpui-pre/src
```

### 1. Inherited layout direction (`style.rs`, `styled.rs`, `window.rs`, `taffy.rs`)

- New `LayoutDirection { Ltr, Rtl }` and a `direction` field on `TextStyle` (so
  `TextStyleRefinement` gets `Option<LayoutDirection>` and it inherits through
  the text-style stack exactly like font settings).
- `Styled::rtl()` / `Styled::ltr()` set the direction and the matching text
  alignment (right / left); `Styled::layout_direction(d)` sets only the direction.
- `Window::request_layout` / `request_measured_layout` read the inherited
  `window.text_style().direction` and pass it to `TaffyLayoutEngine`, which
  sets taffy's `Style::direction`. Taffy 0.13 then mirrors flex rows, grid
  columns and cross-axis alignment. Paddings, margins and absolute insets stay
  physical, as in CSS.
- `LayoutDirection::set_default(d)` (a process-wide atomic) is the root of the
  stack: `TextStyle::default()` starts from it, including its text alignment,
  so every window and every overlay (dialogs, popovers, menus, notifications,
  which gpui-kit's `Root` renders outside the app view) is mirrored without
  touching gpui-kit. The app calls it from `rtl::apply` at startup and when the
  RTL setting changes, then refreshes windows.

### 1b. Logical margins, paddings and insets (`taffy.rs`, `apply_direction`)

In RTL, `apply_direction` swaps left/right margins, paddings and insets
(relative *and* absolute) before handing the style to taffy, so they act as
inline-start/inline-end like CSS logical properties. Without it, components
written for LTR break when their row is mirrored: gpui-kit's `Switch` nudges
its thumb with a relative `left(thumb_x)`, so under RTL the thumb started at
the right end of the track and was pushed further right, outside it; `ml_auto`
spacers collapsed toward their siblings; a notification's absolutely placed
icon (`left_4`) overlapped text whose padding had moved. Window-anchored
overlays (`anchored()`) position through element offsets, not insets, and are
unaffected. Borders and corner radii stay physical, since they are painted
from the element style rather than laid out by taffy (so `Sidebar::side` must
be set to `Side::Right` in RTL to keep its border on the content side).

Audit of gpui-component 0.8.0 for physical offsets that assume LTR travel:

| Component | Pattern | Result under RTL with this patch |
| --- | --- | --- |
| Switch | relative `left(thumb_x)` | fixed: thumb mirrors, stays in the track |
| Notification | absolute `left_4` icon + `pl_6` text, absolute `right_1` close | mirrored together, consistent |
| Dialog close button | absolute `top/right` | moves to the top-left corner (correct for RTL) |
| Checkbox | `left_px` check nudge | mirrored 1 px, harmless |
| Progress | absolute `left(start)/right(end)` fill | fills from the right (RTL progress) |
| Tabs indicator | symmetric `left(-c)/right(-c)` or `left_0/right_0` | unchanged |
| Title bar | `left_0` | unchanged in practice (not used on Linux) |
| Scrollbar | painted from bounds | stays on the physical right |
| Slider | absolute `left(position).ml(-8)` thumb and fill, but pointer math in physical x | **drawn mirrored while dragging is physical**: wrap a Slider in `.ltr()` (the app uses none) |
| Resizable / dock | resize handles use `left(neg_offset)` | mirrored; the app uses none |
| Select, Input clear/suffix | flex children, no insets | mirror with the row |

### 2. RTL-aware line wrapping (`text_system/line_layout.rs`)

Upstream wraps a line by cutting its *shaped* glyph run at x positions, which
assumes visual order equals logical order. For bidi text that cuts rows out of
order (row 1 shows the end of the sentence, rows overflow the box). When a
wrapped line contains RTL characters, `layout_wrapped_line` now calls
`layout_rtl_wrapped`, which:

1. finds break opportunities (after spaces) in *logical* order and fills rows
   greedily by the shaped width of each candidate row;
2. shapes each row on its own (so the bidi algorithm orders that row);
3. concatenates the rows end to end into one `LineLayout`, sorting each row's
   glyphs by x so the row's first glyph in iteration order is its leftmost one,
   and emits a `WrapBoundary` at the first glyph of every row after the first.

The existing painting, alignment and hit-testing code then works unchanged
(`aligned_origin_x` measures a row from boundary to boundary). LTR-only text
takes the original path.

### 3. RTL paragraph base for mixed lines (`text_system/line_layout.rs`)

The shaper picks a paragraph direction from the first strong character, so in
an RTL app a Persian sentence that starts with "Ollama" was laid out as an LTR
paragraph (its final period jumped to the wrong end). When the default
direction is RTL and a line contains RTL characters but starts with a Latin
letter, and RTL letters outnumber Latin ones (so a code line holding a
Persian string stays LTR), `shape_line_with_base_direction` shapes it behind a zero-width
U+200F RIGHT-TO-LEFT MARK and removes that mark's glyph and byte offset
afterwards. Lines without RTL characters (URLs, code) keep LTR.

The same mark also works around an upstream shaping bug (in
`gpui-pre-wgpu`'s cosmic-text backend, not vendored): an RTL paragraph that
*starts with a number* lost the number's glyphs entirely ("۱۲ دقیقه پیش" was
shaped as "دقیقه پیش", with the width of the number missing too). So the mark
is added whenever a line holds RTL text but does not open with an RTL letter,
in RTL mode always, in LTR mode when the line's first letter is RTL.

### 4. Bidi-aware caret mapping (`text_system/line_layout.rs`)

`LineLayout::x_for_index`, `index_for_x` and `closest_index_for_x` walked
glyphs in index order and assumed x grows with the byte index. In an RTL run
it falls, so every caret position collapsed to the left end. When a line has
any glyph whose byte index decreases toward its visual neighbour (an RTL run),
these now use `bidi_cells()`: the glyphs sorted by x with their left/right
edges and a per-glyph direction. The caret before a character sits on the
glyph's right edge when the glyph is RTL, on its left edge otherwise; the end
of the line is the trailing edge of the last logical glyph. Pure LTR lines
keep the original code path.

## gpui-base 0.8.0 and gpui-component 0.8.0 (`vendor/gpui-base`, `vendor/gpui-component`)

Copies of the crates.io sources, patched in `[patch.crates-io]`, so text
inputs follow the layout direction. Marked `[seda patch]`.

- `gpui-component/src/input/input.rs` (`Input::render`): with no explicit
  `text_align` on the element, use the inherited `window.text_style().text_align`
  instead of `Left`. The inherited alignment follows the direction, so every
  `Input` and `Textarea` (which renders through `Input` and forwards its style)
  is right-aligned in RTL, including the placeholder; a `.ltr()` container
  (URLs, paths) gets left alignment.
- `gpui-base/src/input/base/state.rs` (`set_text_align`): accept alignment for
  multi-line inputs too (it was single-line only).
- `gpui-base/src/input/editor/display_map/text_wrapper.rs` and
  `.../input/base/element.rs`: the editor shapes and paints each soft-wrapped
  row on its own, aligned within the content width, but placed carets, hit
  tests and selection rectangles with one offset computed from the *longest*
  row. A new `LineLayout::row_offset(row)` gives each row its own offset, used
  by `position_for_index`, `closest_index_for_x`, `wrapped_line_at` and
  `layout_range_corners`.

Regenerate these diffs with `diff -ru ~/.cargo/registry/src/index.crates.io-*/gpui-base-0.8.0/src vendor/gpui-base/src`
(and the same for gpui-component).

### Re-applying on a gpui bump

Copy the new `gpui-pre-X.Y.Z` from `~/.cargo/registry/src` over
`vendor/gpui-pre`, bump the version in `app/Cargo.toml`'s patch if needed, and
re-apply the hunks above (they touch small, stable functions:
`Window::request_layout`, `Window::request_measured_layout`,
`TaffyLayoutEngine::request_layout`/`request_measured_layout`, `TextStyle`,
`Styled`, `LineLayoutCache::layout_wrapped_line`/`layout_line`).

### Known limits

- Caret movement with the arrow keys is logical (Right moves to the next
  byte), so in RTL text the Left/Right keys move the caret the "wrong" visual
  way, as in many editors without visual cursor movement.
- gpui-kit's markdown renderer lays inline objects (inline code, links) out
  as separate pieces, so a Persian paragraph containing them can show those
  pieces in LTR order.
- Scrollbars stay on the physical right.
