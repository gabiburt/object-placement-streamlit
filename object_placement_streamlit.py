"""
Streamlit port of the PyQt6 Image Overlay app.

This app exposes a simple web UI built with Streamlit that lets you
overlay an RGBA image onto a background image. You can adjust the
overlay's size, rotation and flips, position it anywhere on the
background and export three synchronized outputs:

1. **Canvas_contour** – the background with the overlay drawn on top
   and a green contour around the overlay to highlight its edges.
2. **Canvas_grey** – the background where only the region under the
   overlay is filled with a grey colour, with the overlay drawn on top.
3. **objects** – the overlay alone on a transparent background.

The three files share the same base filename and are saved into the
folders ``Canvas_contour``, ``Canvas_grey`` and ``objects`` under a
common ``output`` directory. If a filename collision occurs, a numeric
suffix is appended (e.g., ``image.png`` becomes ``image_1.png``).

This module is intended to be run with ``streamlit``:

    streamlit run object_placement_streamlit.py

You can then deploy the app on Streamlit Community Cloud or
other hosting providers that support Python and Streamlit.

Note: The core image-processing routines mirror the behaviour of the
original PyQt6 implementation where reasonable, including use of
Pillow's MaxFilter/MinFilter for contour generation. However, this
web app does not support drag‑and‑drop or real‑time view zooming;
instead you use sliders and input boxes to control overlay size and
position.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Tuple, List

import streamlit as st  # type: ignore
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# Constants matching the desktop app
CONTOUR_RGBA = (0, 255, 0, 255)  # green contour (inner & outer)
CONTOUR_WIDTH_DEFAULT = 6
GREY_BG_RGB = (128, 128, 128)


def add_contour_to_rgba(
    rgba_img: Image.Image,
    contour_width: int = CONTOUR_WIDTH_DEFAULT,
    inner_color: Tuple[int, int, int, int] = CONTOUR_RGBA,
    outer_color: Tuple[int, int, int, int] = CONTOUR_RGBA,
) -> Image.Image:
    """Add an inner and outer contour around the opaque regions of an RGBA image.

    The algorithm matches the PyQt6 implementation: it first expands
    the alpha channel (MaxFilter) to create the outer contour, then
    expands and erodes (MaxFilter followed by MinFilter) to create
    the inner contour. These two layers are drawn with different
    colours and composited behind the original image.

    Args:
        rgba_img: An RGBA image.
        contour_width: Thickness of the contour in pixels.
        inner_color: RGBA tuple for the inner contour.
        outer_color: RGBA tuple for the outer contour.

    Returns:
        A new RGBA image with the contour applied.
    """
    if contour_width <= 0:
        return rgba_img.copy()

    # Split out the alpha channel; Pillow uses 0=transparent, 255=opaque.
    alpha = rgba_img.split()[-1]

    # Outer contour: dilate the alpha mask.
    outer = alpha.filter(ImageFilter.MaxFilter(contour_width * 2 + 1))
    outer_img = Image.new("RGBA", rgba_img.size, outer_color)
    outer_img.putalpha(outer)

    # Inner contour: dilate then erode (closing operation).
    inner = alpha.filter(ImageFilter.MaxFilter(contour_width * 2 + 1))
    inner = inner.filter(ImageFilter.MinFilter(contour_width * 2 + 1))
    inner_img = Image.new("RGBA", rgba_img.size, inner_color)
    inner_img.putalpha(inner)

    # Composite: outer → inner → original
    base = Image.new("RGBA", rgba_img.size, (0, 0, 0, 0))
    base = Image.alpha_composite(base, outer_img)
    base = Image.alpha_composite(base, inner_img)
    base = Image.alpha_composite(base, rgba_img)
    return base


def process_overlay(
    overlay: Image.Image,
    target_w: int,
    angle_deg: float = 0.0,
    flip_h: bool = False,
    flip_v: bool = False,
    contour: bool = False,
    contour_width: int = CONTOUR_WIDTH_DEFAULT,
    inner_color: Tuple[int, int, int, int] = CONTOUR_RGBA,
    outer_color: Tuple[int, int, int, int] = CONTOUR_RGBA,
) -> Image.Image:
    """Return a new PIL image representing the overlay transformed and scaled.

    The operation order is flips → rotation → contour → scaling. This
    approximates the behaviour of the desktop app but is simplified for
    clarity. The resulting image preserves transparency.

    Args:
        overlay: The original overlay as an RGBA image.
        target_w: Desired width of the result; height is computed from
            the aspect ratio.
        angle_deg: Clockwise rotation in degrees. Positive values
            rotate clockwise, negative values rotate counter‑clockwise.
        flip_h: If True, flip horizontally before rotation.
        flip_v: If True, flip vertically before rotation.
        contour: Whether to apply the green contour.
        contour_width: Thickness of the contour in pixels.
        inner_color: RGBA for the inner contour.
        outer_color: RGBA for the outer contour.

    Returns:
        A new RGBA image.
    """
    img = overlay.copy()

    # Flips
    if flip_h:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_v:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # Rotation: Pillow rotates counter‑clockwise for positive angles.
    if abs(angle_deg) > 1e-6:
        img = img.rotate(-angle_deg, expand=True, resample=Image.BICUBIC)

    # Contour before scaling to avoid aliasing; apply only when requested.
    if contour:
        img = add_contour_to_rgba(img, contour_width, inner_color, outer_color)

    # Scale to target width while preserving aspect ratio.
    w, h = img.size
    if w != 0:
        aspect = h / w
        new_w = max(1, int(target_w))
        new_h = max(1, int(round(new_w * aspect)))
        img = img.resize((new_w, new_h), resample=Image.LANCZOS)
    return img


def composite_images(
    background: Image.Image,
    overlay: Image.Image,
    pos_x: int,
    pos_y: int,
    grey_bg_rgb: Tuple[int, int, int] = GREY_BG_RGB,
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """Generate the three output variants.

    Args:
        background: The background image (RGB or RGBA).
        overlay: The processed overlay image (must be RGBA).
        pos_x: X coordinate where the overlay's left edge will be drawn.
        pos_y: Y coordinate where the overlay's top edge will be drawn.
        grey_bg_rgb: The colour to use for the grey canvas.

    Returns:
        A tuple ``(canvas_contour, canvas_grey, object_only)`` where each is an RGBA
        image matching the background's size for the first two and the overlay's
        size for the third.
    """
    # Ensure background has an alpha channel for compositing.
    bg = background.convert("RGBA")

    # Initialise canvases
    size = bg.size
    canvas_contour = Image.new("RGBA", size, (0, 0, 0, 0))
    canvas_grey = Image.new("RGBA", size, (0, 0, 0, 0))

    # Draw background on both canvases
    canvas_contour.paste(bg, (0, 0))
    canvas_grey.paste(bg, (0, 0))

    # Canvas_contour: overlay (with contour) already provided
    canvas_contour.paste(overlay, (pos_x, pos_y), overlay)

    # Canvas_grey: fill overlay region with grey and then draw overlay without contour
    grey_layer = Image.new("RGBA", overlay.size, (*grey_bg_rgb, 255))
    # Use the alpha channel of the overlay to restrict the grey fill
    canvas_grey.paste(grey_layer, (pos_x, pos_y), overlay.split()[-1])
    canvas_grey.paste(overlay, (pos_x, pos_y), overlay)

    # object only
    object_only = overlay.copy()

    return canvas_contour, canvas_grey, object_only


def coordinated_unique_name(base_name: str, dirs: List[Path]) -> str:
    """Return a filename that does not exist in any of the given directories.

    If ``base_name`` already exists in one or more directories, append
    ``_N`` before the file extension where N is the smallest integer
    starting from 1 that avoids collisions. The logic matches the
    desktop app's behaviour.
    """
    stem = Path(base_name).stem
    suffix = Path(base_name).suffix

    def exists_in_any(name: str) -> bool:
        for d in dirs:
            if (d / name).exists():
                return True
        return False

    candidate = f"{stem}{suffix}"
    if not exists_in_any(candidate):
        return candidate

    i = 1
    while True:
        candidate = f"{stem}_{i}{suffix}"
        if not exists_in_any(candidate):
            return candidate
        i += 1


def main():
    """Entrypoint for Streamlit app."""
    st.set_page_config(page_title="Image Overlay", layout="wide")
    st.title("Image Overlay Web App")

    st.markdown(
        """
        **Instructions**

        1. Upload a background image (PNG, JPG, etc.).
        2. Upload an RGBA overlay (transparent PNG recommended).
        3. Use the controls below to adjust the overlay's size, rotation, flips
           and position.
        4. When you are happy with the preview, click **Save outputs** to
           generate three files: *Canvas_contour*, *Canvas_grey* and
           *objects*.

        The outputs are stored in an ``output`` folder relative to this script.
        """
    )

    # --- File upload ---
    bg_file = st.file_uploader(
        "Background image",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        key="bg",
    )
    ov_file = st.file_uploader(
        "Overlay image",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        key="ov",
    )

    if bg_file is not None and ov_file is not None:
        # Load images into PIL
        bg_img = Image.open(bg_file).convert("RGBA")
        ov_img_orig = Image.open(ov_file).convert("RGBA")

        # Compute slider ranges
        bg_width, bg_height = bg_img.size

        st.sidebar.header("Overlay adjustments")
        # Desired width (pixels). Limit between 8 and min(bg_width, 2×ov_width)
        max_w_default = min(bg_width, ov_img_orig.width * 2)
        default_w = min(bg_width // 4, max_w_default)
        overlay_width = st.sidebar.slider(
            "Overlay width (px)",
            min_value=8,
            max_value=int(max_w_default),
            value=int(default_w),
            step=1,
        )
        # Position controls
        pos_x = st.sidebar.slider(
            "Overlay X position",
            min_value=0,
            max_value=bg_width,
            value=0,
            step=1,
        )
        pos_y = st.sidebar.slider(
            "Overlay Y position",
            min_value=0,
            max_value=bg_height,
            value=0,
            step=1,
        )
        # Rotation
        angle = st.sidebar.slider(
            "Rotation (degrees)",
            min_value=-180,
            max_value=180,
            value=0,
            step=1,
        )
        # Flips
        flip_h = st.sidebar.checkbox("Flip horizontally", value=False)
        flip_v = st.sidebar.checkbox("Flip vertically", value=False)

        # Note: For simplicity, we allow the overlay to extend beyond the background; it will be clipped.
        # Process overlay without contour (for preview) and with contour (for saving)
        ov_processed = process_overlay(
            overlay=ov_img_orig,
            target_w=overlay_width,
            angle_deg=float(angle),
            flip_h=flip_h,
            flip_v=flip_v,
            contour=False,
        )
        ov_with_contour = process_overlay(
            overlay=ov_img_orig,
            target_w=overlay_width,
            angle_deg=float(angle),
            flip_h=flip_h,
            flip_v=flip_v,
            contour=True,
            contour_width=CONTOUR_WIDTH_DEFAULT,
            inner_color=CONTOUR_RGBA,
            outer_color=CONTOUR_RGBA,
        )

        # Clip position if overlay extends beyond background
        ov_w, ov_h = ov_processed.size
        pos_x_clamped = max(0, min(pos_x, bg_width - ov_w))
        pos_y_clamped = max(0, min(pos_y, bg_height - ov_h))

        # Compose preview: draw plain overlay over background
        preview = bg_img.copy()
        preview.paste(ov_processed, (pos_x_clamped, pos_y_clamped), ov_processed)

        # Display preview
        st.subheader("Preview")
        st.image(preview, caption="Live preview", use_column_width=True)

        # Save outputs
        if st.button("Save outputs"):
            # Create output directories
            root = Path.cwd() / "output"
            canvas_contour_dir = root / "Canvas_contour"
            canvas_grey_dir = root / "Canvas_grey"
            objects_dir = root / "objects"
            for d in (canvas_contour_dir, canvas_grey_dir, objects_dir):
                d.mkdir(parents=True, exist_ok=True)

            base_name = Path(bg_file.name).name
            out_name = coordinated_unique_name(
                base_name,
                [canvas_contour_dir, canvas_grey_dir, objects_dir],
            )

            # Generate variants
            canvas_contour, canvas_grey, object_only = composite_images(
                background=bg_img,
                overlay=ov_with_contour,
                pos_x=pos_x_clamped,
                pos_y=pos_y_clamped,
                grey_bg_rgb=GREY_BG_RGB,
            )

            # Save files
            canvas_contour_path = canvas_contour_dir / out_name
            canvas_grey_path = canvas_grey_dir / out_name
            objects_path = objects_dir / out_name
            canvas_contour.save(canvas_contour_path)
            canvas_grey.save(canvas_grey_path)
            object_only.save(objects_path)

            st.success(
                f"Saved files:\n"
                f"- Canvas_contour → {canvas_contour_path}\n"
                f"- Canvas_grey → {canvas_grey_path}\n"
                f"- objects → {objects_path}"
            )

            # Provide download buttons
            with open(canvas_contour_path, "rb") as fcc:
                st.download_button(
                    label=f"Download Canvas_contour ({out_name})",
                    data=fcc.read(),
                    file_name=out_name,
                    mime="image/png",
                )
            with open(canvas_grey_path, "rb") as fcg:
                st.download_button(
                    label=f"Download Canvas_grey ({out_name})",
                    data=fcg.read(),
                    file_name=out_name,
                    mime="image/png",
                )
            with open(objects_path, "rb") as fobj:
                st.download_button(
                    label=f"Download objects ({out_name})",
                    data=fobj.read(),
                    file_name=out_name,
                    mime="image/png",
                )


if __name__ == "__main__":
    main()