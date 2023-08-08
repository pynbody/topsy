from io import BytesIO

from matplotlib.figure import Figure
import matplotlib.pyplot as plt

def text_to_rgba(s, *, dpi, **kwargs):
    """Render text to RGBA image.

    Based on
    https://matplotlib.org/stable/gallery/text_labels_and_annotations/mathtext_asarray.html"""

    fig = Figure(facecolor="none")
    fig.text(0, 0, s, **kwargs)
    with BytesIO() as buf:
        fig.savefig(buf, dpi=dpi, format="png", bbox_inches="tight",
                    pad_inches=0)
        buf.seek(0)
        rgba = plt.imread(buf)
    return rgba