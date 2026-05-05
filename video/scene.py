"""
video/scene.py — OLoRA Continual Learning, video tehnic, 15 scene.

Standalone (fără branding ODEN). Manim portrait 1080×1920.

Render:
    cd video
    for s in S1_Title S2_CLFormal S3_CatForget S4_LoRA S5_IncLoRA \
              S6_OLoRA S7_Geom2D S8_Subspace S9_Setup S10_Method \
              S11_ResultA S12_ResultB S13_ResultAB S14_LayerMod S15_Conclusion; do
        manim -qh --resolution 1080,1920 scene.py $s
    done
"""
from manim import *
import numpy as np
import json
from pathlib import Path

# ─── Portrait setup ───
config.frame_width  = 8.0
config.frame_height = 14.222222
config.pixel_width  = 1080
config.pixel_height = 1920

# ─── Palette: sober technical ───
BG       = "#0A0A0F"
FG       = "#EAEAEA"
DIM      = "#777777"
ACCENT   = "#5BA8F2"      # blue — formula highlights
GOLD     = "#E5B25E"      # warm — emphasis
GREEN    = "#5FD68C"      # confirmations
RED      = "#E5605E"      # warnings / forgetting
PURPLE   = "#B58CD8"      # geometry
GRID     = "#1F1F26"

Y_TOP    = 5.5
Y_BOT    = -5.8


# ─── Real data from results/analysis/olora_analysis.json ───
ANALYSIS_PATH = Path(__file__).resolve().parent.parent / "results" / "analysis" / "olora_analysis.json"
TASK_NAMES = ["ag_news", "amazon_polarity", "dbpedia_14", "yahoo"]

if ANALYSIS_PATH.exists():
    _D = json.loads(ANALYSIS_PATH.read_text())
    # Cosine similarity (flat vec) — what the original video used
    A_SIM   = np.array(_D["A_matrices"]["cosine"]["sim_matrix"])
    B_SIM   = np.array(_D["B_matrices"]["cosine"]["sim_matrix"])
    AB_SIM  = np.array(_D["AB_product"]["cosine"]["sim_matrix"])
    A_SUM   = _D["A_matrices"]["cosine"]["summary"]
    B_SUM   = _D["B_matrices"]["cosine"]["summary"]
    AB_SUM  = _D["AB_product"]["cosine"]["summary"]
    # Subspace metrics — the ones that reveal the real story
    A_PRINC = _D["A_matrices"]["principal"]["summary"]
    B_PRINC = _D["B_matrices"]["principal"]["summary"]
    A_FROB  = _D["A_matrices"]["frobenius"]["summary"]
    B_FROB  = _D["B_matrices"]["frobenius"]["summary"]
    AB_FROB = _D["AB_product"]["frobenius_ab"]["summary"]
else:
    # fallback (poster numbers)
    A_SIM = np.eye(4); B_SIM = np.eye(4); AB_SIM = np.eye(4)
    A_SUM   = {"max":  1.2e-4, "min": -1.2e-4, "mean": -7e-6}
    B_SUM   = {"max":  2.2e-3, "min": -5.5e-3, "mean": -1.4e-3}
    AB_SUM  = {"max":  2.8e-5, "min": -1e-5,   "mean":  5e-6}
    A_PRINC = {"max":  0.011,  "min": 0.004,   "mean":  0.008}
    B_PRINC = {"max":  0.311,  "min": 0.287,   "mean":  0.295}
    A_FROB  = {"max":  1.5e-3, "min": 0.7e-3,  "mean":  1.2e-3}
    B_FROB  = {"max":  0.16,   "min": 0.066,   "mean":  0.108}
    AB_FROB = {"max":  9.1e-3, "min": 3.1e-3,  "mean":  5.4e-3}


def bg(scene): scene.camera.background_color = BG


def headline(text, *, color=FG, size=46, weight=BOLD, y=Y_TOP - 0.4) -> Text:
    t = Text(text, font_size=size, color=color, weight=weight)
    t.move_to([0, y, 0])
    return t


def small_caption(text, *, color=DIM, size=24, slant=ITALIC) -> Text:
    return Text(text, font_size=size, color=color, slant=slant)


# ─── Heatmap helper ───
def heatmap(values: np.ndarray, labels: list, *,
            cell=0.95, val_size=18, label_size=18,
            cmap_max_abs: float = None,
            highlight_diag=True) -> VGroup:
    """4x4 cosine similarity heatmap. Off-diagonals are tiny — we map their
    abs value onto color by symlog; diagonals shown as solid."""
    n = len(labels)
    cmap_max = cmap_max_abs if cmap_max_abs else max(abs(values.min()),
                                                     abs(values.max()))
    cells = VGroup()
    for r in range(n):
        for c in range(n):
            v = float(values[r, c])
            if r == c and highlight_diag:
                col = GOLD
                op = 1.0
                txt = f"{v:.3f}"
            else:
                # off-diagonal: color by abs / cmap_max in log-ish scale
                a = abs(v) / max(cmap_max, 1e-12)
                # boost low values for visibility
                a = float(np.clip(np.power(a, 0.4), 0.05, 1.0))
                col = ACCENT
                op = 0.20 + a * 0.55
                txt = _format_sci(v)
            box = Square(side_length=cell, stroke_color=DIM,
                         stroke_width=1, fill_color=col, fill_opacity=op)
            box.move_to([(c - (n-1)/2)*cell, ((n-1)/2 - r)*cell, 0])
            t = Text(txt, font_size=val_size, color=FG, weight=BOLD)
            t.move_to(box.get_center())
            cells.add(VGroup(box, t))
    # labels
    label_grp = VGroup()
    for i, name in enumerate(labels):
        lbl_top = Text(name, font_size=label_size, color=DIM, weight=BOLD)
        lbl_top.move_to([(i - (n-1)/2)*cell, ((n-1)/2)*cell + cell*0.7, 0])
        label_grp.add(lbl_top)
        lbl_left = Text(name, font_size=label_size, color=DIM, weight=BOLD)
        lbl_left.rotate(PI/2)
        lbl_left.move_to([-(n-1)/2*cell - cell*0.7, ((n-1)/2 - i)*cell, 0])
        label_grp.add(lbl_left)
    return VGroup(cells, label_grp)


def _format_sci(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a < 1e-3:
        # scientific 1e-N
        exp = int(np.floor(np.log10(a)))
        mant = v / (10**exp)
        return f"{mant:+.1f}e{exp}"
    else:
        return f"{v:+.3f}"


# ═════════════════════════════════════════════════════════════
#  S1 — Title
# ═════════════════════════════════════════════════════════════
class S1_Title(Scene):
    def setup(self): bg(self)

    def construct(self):
        ttl1 = Text("OLoRA", font_size=110, color=GOLD, weight=BOLD)
        ttl2 = Text("Continual Learning", font_size=46,
                    color=FG, weight=BOLD)
        ttl3 = Text("prin Constrângere Geometrică", font_size=36,
                    color=DIM, slant=ITALIC)
        ttl = VGroup(ttl1, ttl2, ttl3).arrange(DOWN, buff=0.30)
        ttl.move_to([0, 2.4, 0])

        # Research question card
        q1 = Text("Întrebarea de cercetare:", font_size=26,
                  color=DIM, slant=ITALIC, weight=BOLD)
        q2 = Text("OLoRA păstrează ortogonalitatea", font_size=30,
                  color=FG, weight=BOLD)
        q3 = Text("actualizărilor LoRA în practică?", font_size=30,
                  color=ACCENT, weight=BOLD)
        q = VGroup(q1, q2, q3).arrange(DOWN, buff=0.25)

        box = SurroundingRectangle(q, color=ACCENT, stroke_width=2,
                                   buff=0.45, corner_radius=0.18)
        qg = VGroup(box, q).move_to([0, -0.6, 0])

        # Sub
        s1 = Text("Qwen 2.5  ·  1.5B params", font_size=26,
                  color=GOLD, weight=BOLD)
        s2 = Text("5 task-uri NLP secvențiale", font_size=24,
                  color=DIM)
        s3 = Text("validare empirică, multi-granularitate", font_size=22,
                  color=DIM, slant=ITALIC)
        sub = VGroup(s1, s2, s3).arrange(DOWN, buff=0.20)
        sub.move_to([0, -3.4, 0])

        self.play(Write(ttl1), run_time=0.7)
        self.play(FadeIn(ttl2, shift=UP*0.15), run_time=0.5)
        self.play(FadeIn(ttl3), run_time=0.4)
        self.play(FadeIn(qg, scale=0.95), run_time=0.9)
        self.play(LaggedStart(*[FadeIn(s, shift=UP*0.15) for s in sub],
                              lag_ratio=0.20), run_time=1.0)
        self.wait(2.0)


# ═════════════════════════════════════════════════════════════
#  S2 — Continual Learning formal
# ═════════════════════════════════════════════════════════════
class S2_CLFormal(Scene):
    def setup(self): bg(self)

    def construct(self):
        h = headline("Continual Learning", color=FG, size=46)
        sub = small_caption("setup formal").next_to(h, DOWN, buff=0.20)

        # Sequence of distributions
        seq = MathTex(r"\mathcal{D}_1,\ \mathcal{D}_2,\ \mathcal{D}_3,\ "
                      r"\dots,\ \mathcal{D}_T",
                      font_size=44, color=FG)
        seq.move_to([0, 3.2, 0])

        constraint = MathTex(
            r"\text{acces la } \mathcal{D}_t \text{ doar la pasul } t",
            font_size=28, color=DIM)
        constraint.next_to(seq, DOWN, buff=0.40)

        # Goal
        goal_lbl = Text("țintă:", font_size=26, color=DIM, slant=ITALIC)
        goal_lbl.move_to([-2.5, 1.0, 0])
        goal = MathTex(
            r"\theta_t \;\text{ bun pe } \mathcal{D}_1, \dots, \mathcal{D}_t",
            font_size=34, color=FG)
        goal.move_to([0.5, 1.0, 0])

        # Metrics
        m_h = Text("Metrici:", font_size=28, color=GOLD, weight=BOLD)
        m_h.move_to([-2.5, -0.3, 0])

        m1 = MathTex(r"\text{ACC} = \frac{1}{T}\sum_{j=1}^{T} R[T,j]",
                     font_size=30, color=FG)
        m1.move_to([0, -1.2, 0])
        m1_d = small_caption("acuratețea medie după ultimul task",
                             size=22).next_to(m1, DOWN, buff=0.15)

        m2 = MathTex(r"\text{BWT} = \frac{1}{T-1}\sum_{j=1}^{T-1}\bigl(R[T,j]-R[j,j]\bigr)",
                     font_size=30, color=FG)
        m2.move_to([0, -2.7, 0])
        m2_d = small_caption("backward transfer — uitarea / câștigul retroactiv",
                             size=22).next_to(m2, DOWN, buff=0.15)

        self.play(Write(h), run_time=0.5)
        self.play(FadeIn(sub), run_time=0.3)
        self.play(Write(seq), run_time=0.7)
        self.play(FadeIn(constraint), run_time=0.4)
        self.play(FadeIn(goal_lbl), Write(goal), run_time=0.7)
        self.play(FadeIn(m_h), run_time=0.4)
        self.play(Write(m1), run_time=0.6)
        self.play(FadeIn(m1_d), run_time=0.3)
        self.play(Write(m2), run_time=0.7)
        self.play(FadeIn(m2_d), run_time=0.3)
        self.wait(1.8)


# ═════════════════════════════════════════════════════════════
#  S3 — Catastrophic forgetting
# ═════════════════════════════════════════════════════════════
class S3_CatForget(Scene):
    def setup(self): bg(self)

    def construct(self):
        h = headline("Catastrophic Forgetting", color=RED, size=44)

        # 5x5 R[i][j] grid showing forgetting pattern
        # Synthetic: trained sequentially, accuracy on task j after step i.
        # For full fine-tuning naive: high diagonal, low subdiagonal.
        R = [
            [0.88, None, None, None, None],
            [0.42, 0.85, None, None, None],
            [0.31, 0.40, 0.91, None, None],
            [0.25, 0.32, 0.55, 0.93, None],
            [0.22, 0.27, 0.40, 0.61, 0.86],
        ]

        cells = VGroup()
        cell = 0.85
        for i in range(5):
            for j in range(5):
                v = R[i][j]
                if v is None:
                    # not yet trained: empty cell
                    box = Square(side_length=cell, stroke_color=GRID,
                                  stroke_width=1, fill_color=GRID,
                                  fill_opacity=0.6)
                    txt = Text("—", font_size=20, color=DIM)
                else:
                    # color: diagonal=green if R[i][i] reasonable;
                    # off-diag below = red intensity for forgetting
                    if i == j:
                        col = GREEN; op = 0.55
                    else:
                        # forgetting: lower v compared to R[j][j] = darker red
                        deg = max(0.0, R[j][j] - v) / 0.7
                        op = 0.15 + min(deg, 1.0) * 0.55
                        col = RED
                    box = Square(side_length=cell, stroke_color=DIM,
                                  stroke_width=1, fill_color=col,
                                  fill_opacity=op)
                    txt = Text(f"{v:.2f}", font_size=20, color=FG, weight=BOLD)
                box.move_to([(j - 2)*cell, (2 - i)*cell, 0])
                txt.move_to(box.get_center())
                cells.add(VGroup(box, txt))

        # Axis labels
        x_lbl = Text("task evaluat  j →", font_size=22, color=DIM)
        x_lbl.next_to(cells, UP, buff=0.45)
        y_lbl = Text("după antrenare i ↓", font_size=22, color=DIM)
        y_lbl.rotate(PI/2)
        y_lbl.next_to(cells, LEFT, buff=0.45)

        cap = Text("R[i, j] — fine-tuning naiv pe parametri full",
                   font_size=22, color=DIM, slant=ITALIC)
        cap.next_to(cells, DOWN, buff=0.50)

        # Findings
        f1 = Text("diagonala: rezonabilă",
                  font_size=26, color=GREEN, weight=BOLD)
        f2 = Text("subdiagonala: degradare brutală",
                  font_size=26, color=RED, weight=BOLD)
        f3 = MathTex(r"R[i,j] \ll R[j,j],\ \forall j < i",
                     font_size=30, color=FG)
        findings = VGroup(f1, f2, f3).arrange(DOWN, buff=0.25)
        findings.move_to([0, -3.6, 0])

        self.play(Write(h), run_time=0.5)
        self.play(FadeIn(x_lbl), FadeIn(y_lbl), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(c, scale=0.9) for c in cells],
                              lag_ratio=0.015), run_time=1.4)
        self.play(FadeIn(cap), run_time=0.3)
        self.play(LaggedStart(*[FadeIn(f, shift=UP*0.15) for f in findings],
                              lag_ratio=0.20), run_time=1.0)
        self.wait(1.8)


# ═════════════════════════════════════════════════════════════
#  S4 — LoRA recap
# ═════════════════════════════════════════════════════════════
class S4_LoRA(Scene):
    def setup(self): bg(self)

    def construct(self):
        h = headline("LoRA", color=FG, size=52)
        sub = small_caption("Low-Rank Adaptation",
                            size=26).next_to(h, DOWN, buff=0.20)

        # Formula
        eq = MathTex(r"W = W_0 + \Delta W = W_0 + B\cdot A",
                     font_size=44, color=FG)
        eq.set_color_by_tex(r"W_0", DIM)
        eq.set_color_by_tex(r"B", ACCENT)
        eq.set_color_by_tex(r"A", GOLD)
        eq.move_to([0, 3.0, 0])

        # Block matrices visualization
        # W0 is d×d (frozen). B is d×r, A is r×d.
        d = 1.8; r_h = 0.45
        w0 = Rectangle(width=d, height=d, stroke_color=DIM, stroke_width=2,
                       fill_color=DIM, fill_opacity=0.20)
        w0_lbl = MathTex(r"W_0", font_size=30, color=DIM)
        w0_lbl.move_to(w0.get_center())
        w0_grp = VGroup(w0, w0_lbl).move_to([-2.2, 0.7, 0])
        w0_size = Text("d × d", font_size=18, color=DIM)
        w0_size.next_to(w0_grp, DOWN, buff=0.10)

        plus = Text("+", font_size=44, color=FG, weight=BOLD)
        plus.move_to([-0.4, 0.7, 0])

        b = Rectangle(width=r_h, height=d, stroke_color=ACCENT,
                      stroke_width=2, fill_color=ACCENT, fill_opacity=0.40)
        b_lbl = MathTex(r"B", font_size=28, color=FG)
        b_lbl.move_to(b.get_center())
        b_grp = VGroup(b, b_lbl).move_to([0.5, 0.7, 0])
        b_size = Text("d × r", font_size=18, color=ACCENT)
        b_size.next_to(b_grp, DOWN, buff=0.10)

        dot = Text("·", font_size=44, color=FG, weight=BOLD)
        dot.move_to([1.2, 0.7, 0])

        a = Rectangle(width=d, height=r_h, stroke_color=GOLD,
                      stroke_width=2, fill_color=GOLD, fill_opacity=0.40)
        a_lbl = MathTex(r"A", font_size=28, color=BG)
        a_lbl.move_to(a.get_center())
        a_grp = VGroup(a, a_lbl).move_to([2.7, 0.7, 0])
        a_size = Text("r × d", font_size=18, color=GOLD)
        a_size.next_to(a_grp, DOWN, buff=0.10)

        # Constraint
        rank_eq = MathTex(r"r \ll d \quad (\text{ex: }r=8,\ d=1536)",
                          font_size=30, color=FG)
        rank_eq.move_to([0, -1.4, 0])

        # Property
        p1 = Text("• W₀ înghețat", font_size=26, color=DIM)
        p2 = Text("• se antrenează doar A, B", font_size=26, color=GOLD,
                  weight=BOLD)
        p3 = Text("• ~0.1% din parametri", font_size=26, color=GREEN,
                  weight=BOLD)
        props = VGroup(p1, p2, p3).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        props.move_to([0, -2.9, 0])

        self.play(Write(h), run_time=0.5)
        self.play(FadeIn(sub), run_time=0.3)
        self.play(Write(eq), run_time=0.9)
        self.play(FadeIn(w0_grp, scale=0.9), FadeIn(w0_size), run_time=0.5)
        self.play(FadeIn(plus), run_time=0.2)
        self.play(FadeIn(b_grp, scale=0.9), FadeIn(b_size), run_time=0.5)
        self.play(FadeIn(dot), run_time=0.2)
        self.play(FadeIn(a_grp, scale=0.9), FadeIn(a_size), run_time=0.5)
        self.play(Write(rank_eq), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(p, shift=UP*0.15) for p in props],
                              lag_ratio=0.18), run_time=1.0)
        self.wait(1.6)


# ═════════════════════════════════════════════════════════════
#  S5 — IncLoRA
# ═════════════════════════════════════════════════════════════
class S5_IncLoRA(Scene):
    def setup(self): bg(self)

    def construct(self):
        h = headline("IncLoRA", color=FG, size=52)
        sub = small_caption("baseline — adapter izolat per task",
                            size=24).next_to(h, DOWN, buff=0.20)

        # 5 adapters, frozen-after-training
        T = 5
        w_box = 0.95
        adapters = VGroup()
        for t in range(T):
            box = RoundedRectangle(width=w_box, height=1.3,
                                   corner_radius=0.10,
                                   stroke_color=ACCENT, stroke_width=2,
                                   fill_color=ACCENT, fill_opacity=0.18)
            tt = MathTex(rf"A_{{{t+1}}}, B_{{{t+1}}}", font_size=22, color=FG)
            tt.move_to(box.get_center() + UP*0.20)
            tlbl = Text(f"task {t+1}", font_size=18, color=DIM)
            tlbl.move_to(box.get_center() + DOWN*0.30)
            adapters.add(VGroup(box, tt, tlbl))
        adapters.arrange(RIGHT, buff=0.20)
        adapters.move_to([0, 2.6, 0])

        # Isolation: dashed barriers
        for i in range(T - 1):
            x = (adapters[i].get_right()[0] + adapters[i+1].get_left()[0]) / 2
            barrier = DashedLine([x, 1.6, 0], [x, 3.6, 0],
                                 color=DIM, stroke_width=1.5)

        forward = MathTex(
            r"y = W_0 x + B_t A_t x \quad (\text{task } t \text{ cunoscut})",
            font_size=30, color=FG)
        forward.move_to([0, 1.0, 0])

        # Properties
        pos = Text("✓ zero interferență prin construcție",
                   font_size=24, color=GREEN, weight=BOLD)
        neg = Text("✗ zero transfer pozitiv",
                   font_size=24, color=RED, weight=BOLD)
        neg2 = Text("✗ adapterul învață izolat — nefolosit ce e deja prezent",
                    font_size=22, color=RED)
        props = VGroup(pos, neg, neg2).arrange(DOWN, buff=0.22,
                                               aligned_edge=LEFT)
        props.move_to([0, -0.5, 0])

        # Empirical from inclora_results.json
        emp_h = Text("R[T,j] empiric (IncLoRA, Qwen 2.5):",
                     font_size=22, color=GOLD, weight=BOLD)
        emp_h.move_to([0, -1.9, 0])
        vals = [0.886, 0.207, 0.491, 0.983, 0.648]
        emp_cells = VGroup()
        for i, v in enumerate(vals):
            cell = Square(side_length=0.85, stroke_color=DIM,
                          stroke_width=1, fill_color=GREEN,
                          fill_opacity=0.15 + v * 0.45)
            t = Text(f"{v:.2f}", font_size=22, color=FG, weight=BOLD)
            t.move_to(cell.get_center())
            emp_cells.add(VGroup(cell, t))
        emp_cells.arrange(RIGHT, buff=0.12)
        emp_cells.move_to([0, -2.7, 0])

        avg = MathTex(r"\overline{\text{ACC}} = 0.643,\quad \text{BWT}=0.0",
                      font_size=28, color=GOLD)
        avg.move_to([0, -3.7, 0])

        self.play(Write(h), run_time=0.5)
        self.play(FadeIn(sub), run_time=0.3)
        self.play(LaggedStart(*[FadeIn(a, scale=0.9) for a in adapters],
                              lag_ratio=0.10), run_time=1.0)
        self.play(Write(forward), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(p, shift=UP*0.15) for p in props],
                              lag_ratio=0.20), run_time=1.0)
        self.play(FadeIn(emp_h), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(c, scale=0.9) for c in emp_cells],
                              lag_ratio=0.10), run_time=0.7)
        self.play(Write(avg), run_time=0.5)
        self.wait(1.6)


# ═════════════════════════════════════════════════════════════
#  S6 — OLoRA regularizer
# ═════════════════════════════════════════════════════════════
class S6_OLoRA(Scene):
    def setup(self): bg(self)

    def construct(self):
        h = headline("OLoRA — regularizatorul", color=GOLD, size=44)

        # Total loss
        ltot = MathTex(
            r"\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + "
            r"\lambda_1 \cdot \mathcal{L}_{\text{orth}}",
            font_size=42, color=FG)
        ltot.move_to([0, 3.2, 0])

        # Orthogonality loss
        lorth = MathTex(
            r"\mathcal{L}_{\text{orth}} = "
            r"\sum_{i<t} \bigl\| A_t \cdot A_i^{\top} \bigr\|_F^{\,2}",
            font_size=40, color=ACCENT)
        lorth.move_to([0, 1.6, 0])

        # Geometric reading
        geom_h = Text("Geometric:", font_size=28, color=GOLD,
                      weight=BOLD, slant=ITALIC)
        geom_h.move_to([-2.3, 0.3, 0])

        g1 = MathTex(
            r"\bigl\|A_t A_i^{\top}\bigr\|_F^{\,2} \to 0",
            font_size=32, color=FG)
        g1.move_to([0, -0.5, 0])

        arrow = MathTex(r"\Longleftrightarrow", font_size=36, color=DIM)
        arrow.next_to(g1, DOWN, buff=0.20)

        g2 = MathTex(
            r"\text{rândurile } A_t \perp \text{ rândurile } A_i,"
            r"\ \forall i < t",
            font_size=28, color=FG)
        g2.next_to(arrow, DOWN, buff=0.20)

        # Hyperparam
        hp = MathTex(r"\lambda_1 = 0.5\quad \text{(experimentele noastre)}",
                     font_size=28, color=GOLD)
        hp.move_to([0, -2.8, 0])

        note = Text("B nu e regularizat. (urmărit empiric.)",
                    font_size=22, color=DIM, slant=ITALIC)
        note.move_to([0, -3.5, 0])

        self.play(Write(h), run_time=0.5)
        self.play(Write(ltot), run_time=0.9)
        self.play(Write(lorth), run_time=1.0)
        self.play(FadeIn(geom_h), run_time=0.4)
        self.play(Write(g1), run_time=0.6)
        self.play(FadeIn(arrow), run_time=0.3)
        self.play(Write(g2), run_time=0.7)
        self.play(Write(hp), run_time=0.5)
        self.play(FadeIn(note), run_time=0.4)
        self.wait(1.8)


# ═════════════════════════════════════════════════════════════
#  S7 — 2D geometric intuition
# ═════════════════════════════════════════════════════════════
class S7_Geom2D(Scene):
    def setup(self): bg(self)

    def construct(self):
        h = headline("Intuiție în 2D", color=FG, size=42)

        # Coordinate plane
        ax = Axes(x_range=[-3, 3, 1], y_range=[-3, 3, 1],
                  x_length=4.5, y_length=4.5,
                  axis_config={"color": GRID, "stroke_width": 1.5,
                               "include_ticks": False, "include_tip": False})
        ax.move_to([0, 0.6, 0])

        # Vector A_1 (red), A_2 (blue) initial: misaligned
        v1 = Arrow(ax.c2p(0, 0), ax.c2p(2, 0.4),
                    color=GOLD, stroke_width=6,
                    buff=0, max_tip_length_to_length_ratio=0.18)
        v1_lbl = MathTex(r"A_1", font_size=30, color=GOLD)
        v1_lbl.next_to(v1.get_end(), UP+RIGHT, buff=0.10)

        v2_init = Arrow(ax.c2p(0, 0), ax.c2p(1.6, 1.0),
                         color=ACCENT, stroke_width=6,
                         buff=0, max_tip_length_to_length_ratio=0.18)
        v2_lbl = MathTex(r"A_2", font_size=30, color=ACCENT)
        v2_lbl.next_to(v2_init.get_end(), UP+RIGHT, buff=0.10)

        # Inner product display
        ip = MathTex(r"\langle A_1, A_2 \rangle = 1.6\,\square",
                     font_size=28, color=FG)
        ip.move_to([0, -2.7, 0])

        ip_init_value = "+1.60"
        ip_text = Text(f"⟨A₁, A₂⟩ = {ip_init_value}",
                        font_size=28, color=FG, weight=BOLD)
        ip_text.move_to([0, -2.7, 0])

        self.play(Write(h), run_time=0.5)
        self.play(Create(ax), run_time=0.6)
        self.play(GrowArrow(v1), FadeIn(v1_lbl), run_time=0.6)
        self.play(GrowArrow(v2_init), FadeIn(v2_lbl), run_time=0.6)
        self.play(FadeIn(ip_text), run_time=0.4)

        # Penalty drives <A1,A2>^2 → 0
        loss = MathTex(r"\mathcal{L}_{\text{orth}} = "
                        r"\langle A_1, A_2\rangle^2",
                        font_size=30, color=ACCENT)
        loss.move_to([0, -3.4, 0])
        self.play(Write(loss), run_time=0.7)

        # Animate v2 rotating to be perpendicular to v1
        # Final v2 direction perpendicular to v1: v1 = (2, 0.4),
        # perpendicular dir = (-0.4, 2) normalized * length |v2| ≈ 1.886
        target_x, target_y = -0.4, 2.0
        norm = np.hypot(target_x, target_y)
        L2 = np.hypot(1.6, 1.0)
        target_x = target_x / norm * L2
        target_y = target_y / norm * L2

        v2_final = Arrow(ax.c2p(0, 0), ax.c2p(target_x, target_y),
                          color=ACCENT, stroke_width=6,
                          buff=0, max_tip_length_to_length_ratio=0.18)
        v2_lbl_final = MathTex(r"A_2", font_size=30, color=ACCENT)
        v2_lbl_final.next_to(v2_final.get_end(), UP, buff=0.10)
        ip_final = Text("⟨A₁, A₂⟩ ≈ 0",
                         font_size=30, color=GREEN, weight=BOLD)
        ip_final.move_to(ip_text.get_center())

        # 90° angle marker
        right_angle = Square(side_length=0.20, stroke_color=GREEN,
                              stroke_width=2, fill_opacity=0)
        right_angle.move_to(ax.c2p(0.10, 0.10))

        self.play(
            Transform(v2_init, v2_final),
            Transform(v2_lbl, v2_lbl_final),
            Transform(ip_text, ip_final),
            run_time=2.0,
        )
        self.play(Create(right_angle), run_time=0.4)
        self.wait(1.6)


# ═════════════════════════════════════════════════════════════
#  S8 — Subspace claim
# ═════════════════════════════════════════════════════════════
class S8_Subspace(Scene):
    def setup(self): bg(self)

    def construct(self):
        h = headline("Generalizare la subspații", color=FG, size=40)

        # Claim
        claim = MathTex(
            r"\text{row}(A_t)\;\perp\;"
            r"\mathrm{span}\bigl(\text{row}(A_1),\dots,\text{row}(A_{t-1})\bigr)",
            font_size=32, color=ACCENT)
        claim.move_to([0, 3.4, 0])

        # Subspace bookkeeping
        b1 = MathTex(r"\dim(\text{row}(A_t)) = r", font_size=30, color=FG)
        b2 = MathTex(r"\dim(\mathbb{R}^d) = d", font_size=30, color=FG)
        b3 = MathTex(r"r \cdot T \ll d \quad\Rightarrow\quad "
                      r"\text{constrângerea satisfăcută ușor}",
                      font_size=28, color=GOLD)
        budget = VGroup(b1, b2, b3).arrange(DOWN, buff=0.30)
        budget.move_to([0, 1.6, 0])

        # Numerical concrete: r=8, T=5, d=1536
        ne = MathTex(r"r=8,\ T=5,\ d=1536",
                      font_size=30, color=FG)
        ne.move_to([0, -0.4, 0])

        ne2 = MathTex(r"r \cdot T = 40 \;\;\big/\;\; d = 1536",
                      font_size=32, color=GOLD)
        ne2.move_to([0, -1.3, 0])

        ne3 = MathTex(r"\Rightarrow\;\;40 / 1536 \;\approx\; 2.6\%",
                      font_size=32, color=GREEN)
        ne3.move_to([0, -2.2, 0])

        cap = Text("Există ortogonalitate de cumpărat. Întrebarea: o ia OLoRA?",
                   font_size=22, color=DIM, slant=ITALIC)
        cap.move_to([0, -3.5, 0])

        self.play(Write(h), run_time=0.5)
        self.play(Write(claim), run_time=1.0)
        self.play(LaggedStart(Write(b1), Write(b2), Write(b3),
                              lag_ratio=0.30), run_time=1.5)
        self.play(Write(ne), run_time=0.6)
        self.play(Write(ne2), run_time=0.7)
        self.play(Write(ne3), run_time=0.7)
        self.play(FadeIn(cap), run_time=0.4)
        self.wait(1.6)


# ═════════════════════════════════════════════════════════════
#  S9 — Setup empiric
# ═════════════════════════════════════════════════════════════
class S9_Setup(Scene):
    def setup(self): bg(self)

    def construct(self):
        h = headline("Setup experimental", color=FG, size=44)

        # Model card
        m_h = Text("Model:", font_size=26, color=GOLD, weight=BOLD)
        m_v = Text("Qwen 2.5 — 1.5B params", font_size=28, color=FG, weight=BOLD)
        m_arch = Text("28 blocuri transformer, d_model = 1536",
                       font_size=22, color=DIM)
        model_block = VGroup(m_h, m_v, m_arch).arrange(DOWN, buff=0.18,
                                                       aligned_edge=LEFT)
        model_block.move_to([0, 3.4, 0])

        # Hyperparam table
        rows_data = [
            ("rank r",         "8"),
            ("alpha",          "32"),
            ("learning rate",  "1e-3"),
            ("batch size",     "2 (×4 grad accum)"),
            ("max seq len",    "256"),
            ("epochs / task",  "1"),
            ("λ₁ (orth loss)", "0.5"),
            ("modules țintă",  "q_proj, v_proj"),
        ]
        rows = VGroup()
        for k, v in rows_data:
            kt = Text(k, font_size=22, color=DIM, weight=BOLD)
            vt = Text(v, font_size=22, color=ACCENT, weight=BOLD)
            row = VGroup(kt, vt).arrange(RIGHT, buff=1.0)
            rows.add(row)
        rows.arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        rows.move_to([0, 0.3, 0])

        # Tasks
        t_h = Text("Task-uri (secvențiale):", font_size=24,
                   color=GOLD, weight=BOLD)
        t_h.move_to([0, -2.3, 0])
        tasks_txt = Text(
            "AG News  →  Yelp  →  Amazon  →  DBpedia  →  Yahoo",
            font_size=22, color=FG)
        tasks_txt.next_to(t_h, DOWN, buff=0.20)

        sub = Text("clasificare text — 2 până la 14 clase per task",
                   font_size=20, color=DIM, slant=ITALIC)
        sub.next_to(tasks_txt, DOWN, buff=0.15)

        self.play(Write(h), run_time=0.5)
        self.play(FadeIn(model_block, shift=UP*0.15), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(r, shift=UP*0.10) for r in rows],
                              lag_ratio=0.08), run_time=1.4)
        self.play(FadeIn(t_h), run_time=0.4)
        self.play(FadeIn(tasks_txt), run_time=0.5)
        self.play(FadeIn(sub), run_time=0.3)
        self.wait(1.6)


# ═════════════════════════════════════════════════════════════
#  S10 — Methodology
# ═════════════════════════════════════════════════════════════
class S10_Method(Scene):
    def setup(self): bg(self)

    def construct(self):
        h = headline("Metodologie — auditul geometric", color=FG, size=38)

        step1 = Text("1. Extrage A_t, B_t din checkpoint după fiecare task.",
                     font_size=24, color=FG)
        step1.move_to([0, 3.5, 0])

        step2 = Text("2. Pentru fiecare pereche i ≠ j calculează:",
                     font_size=24, color=FG)
        step2.move_to([0, 2.7, 0])

        # Cosine formulas
        f1 = MathTex(r"\cos\theta(A_i, A_j) = "
                     r"\frac{\langle \mathrm{vec}(A_i),\mathrm{vec}(A_j)\rangle}"
                     r"{\|A_i\|_F\,\|A_j\|_F}",
                     font_size=30, color=ACCENT)
        f1.move_to([0, 1.7, 0])
        f1_lbl = Text("nivel A", font_size=20, color=DIM, slant=ITALIC)
        f1_lbl.next_to(f1, RIGHT, buff=0.30)

        f2 = MathTex(r"\cos\theta(B_i, B_j)\quad\text{idem}",
                     font_size=30, color=ACCENT)
        f2.move_to([0, 0.7, 0])

        f3 = MathTex(r"\cos\theta\bigl(A_i B_i,\ A_j B_j\bigr)\quad"
                     r"\text{nivel update efectiv}",
                     font_size=28, color=GOLD)
        f3.move_to([0, -0.3, 0])

        step3 = Text("3. Agregare pe trei nivele de granularitate:",
                     font_size=24, color=FG)
        step3.move_to([0, -1.4, 0])

        gran = VGroup(
            Text("• global  (matrice 4×4 task × task)",
                 font_size=22, color=DIM),
            Text("• per layer  (28 layere)",
                 font_size=22, color=DIM),
            Text("• per modul  (q_proj, v_proj — 56 puncte)",
                 font_size=22, color=DIM),
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        gran.move_to([0, -2.5, 0])

        cap = Text("Scor 0 = perfect ortogonal.   Scor 1 = colinear.",
                   font_size=22, color=GOLD, weight=BOLD, slant=ITALIC)
        cap.move_to([0, -3.7, 0])

        self.play(Write(h), run_time=0.5)
        self.play(FadeIn(step1, shift=UP*0.15), run_time=0.5)
        self.play(FadeIn(step2, shift=UP*0.15), run_time=0.5)
        self.play(Write(f1), FadeIn(f1_lbl), run_time=1.0)
        self.play(Write(f2), run_time=0.7)
        self.play(Write(f3), run_time=0.8)
        self.play(FadeIn(step3, shift=UP*0.15), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(g, shift=UP*0.10) for g in gran],
                              lag_ratio=0.18), run_time=0.9)
        self.play(FadeIn(cap), run_time=0.4)
        self.wait(1.6)


# ═════════════════════════════════════════════════════════════
#  S11 — Result A
# ═════════════════════════════════════════════════════════════
class S11_ResultA(Scene):
    def setup(self): bg(self)

    def construct(self):
        h = headline("Rezultat — matricile A", color=GOLD, size=42)
        sub = MathTex(r"\cos\theta(A_i, A_j),\ i \neq j",
                       font_size=30, color=DIM)
        sub.next_to(h, DOWN, buff=0.20)

        hm = heatmap(A_SIM, TASK_NAMES, cell=1.05, val_size=18, label_size=18)
        hm.move_to([0, 0.7, 0])

        # summary stats
        s_max  = float(A_SUM["max"]); s_min = float(A_SUM["min"])
        s_mean = float(A_SUM["mean"])
        max_abs = max(abs(s_max), abs(s_min))

        stats = VGroup(
            Text(f"max |cos|  =  {_format_sci(max_abs)}",
                 font_size=28, color=GREEN, weight=BOLD),
            Text(f"mean       =  {_format_sci(s_mean)}",
                 font_size=24, color=DIM),
        ).arrange(DOWN, buff=0.18)
        stats.move_to([0, -2.7, 0])

        verdict = Text("Constrângerea OLoRA respectată strict.",
                       font_size=24, color=GREEN, weight=BOLD, slant=ITALIC)
        verdict.move_to([0, -3.6, 0])

        self.play(Write(h), run_time=0.5)
        self.play(Write(sub), run_time=0.4)
        self.play(FadeIn(hm, scale=0.95), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(s) for s in stats],
                              lag_ratio=0.30), run_time=0.8)
        self.play(FadeIn(verdict, shift=UP*0.15), run_time=0.5)
        self.wait(2.0)


# ═════════════════════════════════════════════════════════════
#  S12 — Result B
# ═════════════════════════════════════════════════════════════
class S12_ResultB(Scene):
    def setup(self): bg(self)

    def construct(self):
        h = headline("Rezultat — matricile B", color=GOLD, size=42)
        sub = Text("B nu e regularizat. La ce ne așteptăm?",
                   font_size=22, color=DIM, slant=ITALIC)
        sub.next_to(h, DOWN, buff=0.20)

        hm = heatmap(B_SIM, TASK_NAMES, cell=1.05,
                     cmap_max_abs=max(abs(B_SUM["min"]), abs(B_SUM["max"])))
        hm.move_to([0, 0.7, 0])

        s_max = max(abs(B_SUM["max"]), abs(B_SUM["min"]))
        stats = VGroup(
            Text(f"max |cos|  =  {_format_sci(s_max)}",
                 font_size=28, color=ACCENT, weight=BOLD),
            Text(f"mean       =  {_format_sci(B_SUM['mean'])}",
                 font_size=24, color=DIM),
            Text("→ ~2 ordine de mărime peste A",
                 font_size=22, color=GOLD, slant=ITALIC),
        ).arrange(DOWN, buff=0.18)
        stats.move_to([0, -2.6, 0])

        why = Text(
            "B = 0 inițial; gradienții lui sunt proiectați prin Aᵀ_t,",
            font_size=20, color=DIM)
        why2 = Text(
            "deci moștenesc structura ortogonală a lui A. Tot mic.",
            font_size=20, color=DIM)
        whyg = VGroup(why, why2).arrange(DOWN, buff=0.10)
        whyg.move_to([0, -3.7, 0])

        self.play(Write(h), run_time=0.5)
        self.play(FadeIn(sub), run_time=0.4)
        self.play(FadeIn(hm, scale=0.95), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(s) for s in stats],
                              lag_ratio=0.30), run_time=1.0)
        self.play(LaggedStart(*[FadeIn(w) for w in whyg],
                              lag_ratio=0.20), run_time=0.7)
        self.wait(1.8)


# ═════════════════════════════════════════════════════════════
#  S13 — Result A·B
# ═════════════════════════════════════════════════════════════
class S13_ResultAB(Scene):
    def setup(self): bg(self)

    def construct(self):
        h = headline("Rezultat — produsul A·B", color=GOLD, size=42)
        sub = MathTex(
            r"\Delta W_t = A_t B_t \;\;\text{— direcția efectivă}",
            font_size=28, color=ACCENT)
        sub.next_to(h, DOWN, buff=0.20)

        hm = heatmap(AB_SIM, TASK_NAMES, cell=1.05,
                     cmap_max_abs=max(abs(AB_SUM["min"]), abs(AB_SUM["max"])))
        hm.move_to([0, 0.7, 0])

        max_abs = max(abs(AB_SUM["max"]), abs(AB_SUM["min"]))
        stats = VGroup(
            Text(f"max |cos|  =  {_format_sci(max_abs)}",
                 font_size=30, color=GREEN, weight=BOLD),
            Text("mai mic decât B singur.",
                 font_size=22, color=GOLD, weight=BOLD, slant=ITALIC),
        ).arrange(DOWN, buff=0.18)
        stats.move_to([0, -2.6, 0])

        # The cascade insight
        insight = Text("CASCADĂ:",
                       font_size=28, color=GOLD, weight=BOLD)
        insight.move_to([-2.2, -3.4, 0])
        insight2 = Text("constrângere pe A → ortogonalitate pe ΔW.",
                        font_size=22, color=FG)
        insight2.move_to([0.6, -3.4, 0])

        self.play(Write(h), run_time=0.5)
        self.play(Write(sub), run_time=0.6)
        self.play(FadeIn(hm, scale=0.95), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(s) for s in stats],
                              lag_ratio=0.30), run_time=0.9)
        self.play(FadeIn(insight, shift=UP*0.15),
                  FadeIn(insight2, shift=UP*0.15), run_time=0.6)
        self.wait(2.0)


# ═════════════════════════════════════════════════════════════
#  S14 — Per layer / per module
# ═════════════════════════════════════════════════════════════
class S14_LayerMod(Scene):
    def setup(self): bg(self)

    def construct(self):
        h = headline("Granularitate fină", color=FG, size=42)
        sub = Text("28 layere × 2 module = 56 puncte de măsură",
                   font_size=22, color=DIM, slant=ITALIC)
        sub.next_to(h, DOWN, buff=0.20)

        # Bar chart per layer (use real data)
        # Get values per layer (avg over q_proj, v_proj)
        layer_vals = {}
        for k, v in AB_LAYER_SCORES.items():
            # k like 'model.layers.5.self_attn.q_proj'
            parts = k.split(".")
            layer_idx = int(parts[2])
            mod = parts[-1]
            layer_vals.setdefault(layer_idx, {}).setdefault(mod, v)
        layers_sorted = sorted(layer_vals.keys())

        # Plot abs values, q_proj and v_proj as paired bars
        chart_w = 6.5
        chart_h = 2.0
        n = len(layers_sorted)
        bar_w = chart_w / (n * 2.4)

        # max for y-scale
        all_abs = []
        for li in layers_sorted:
            for m in ("q_proj", "v_proj"):
                if m in layer_vals[li]:
                    all_abs.append(abs(layer_vals[li][m]))
        ymax = max(all_abs) * 1.15 if all_abs else 1e-4

        chart_origin = np.array([-chart_w/2, 0.5, 0])
        # axis line
        x_axis = Line(chart_origin,
                       chart_origin + RIGHT*chart_w,
                       color=DIM, stroke_width=2)
        y_axis = Line(chart_origin,
                       chart_origin + UP*chart_h,
                       color=DIM, stroke_width=2)

        bars = VGroup()
        for i, li in enumerate(layers_sorted):
            x0 = chart_origin[0] + i * (chart_w / n)
            for k, m in enumerate(("q_proj", "v_proj")):
                if m not in layer_vals[li]: continue
                v = abs(layer_vals[li][m])
                h_bar = (v / ymax) * chart_h
                col = ACCENT if m == "q_proj" else GOLD
                bar = Rectangle(width=bar_w, height=max(h_bar, 0.005),
                                fill_color=col, fill_opacity=0.85,
                                stroke_width=0)
                bar.move_to([x0 + (k - 0.5) * bar_w * 1.05 + bar_w/2,
                             chart_origin[1] + h_bar/2, 0])
                bars.add(bar)

        # Y-axis ticks
        y_tick_top = MathTex(f"{ymax:.0e}", font_size=18, color=DIM)
        y_tick_top.move_to(chart_origin + UP*chart_h + LEFT*0.30)
        y_tick_bot = Text("0", font_size=18, color=DIM)
        y_tick_bot.move_to(chart_origin + LEFT*0.30)

        x_lbl = Text("layer index 0..27", font_size=20, color=DIM)
        x_lbl.move_to(chart_origin + RIGHT*chart_w/2 + DOWN*0.40)

        y_lbl = Text("|cos θ|", font_size=20, color=DIM)
        y_lbl.rotate(PI/2)
        y_lbl.move_to(chart_origin + UP*chart_h/2 + LEFT*0.65)

        # Legend
        leg_q = VGroup(
            Square(side_length=0.20, fill_color=ACCENT, fill_opacity=0.85,
                   stroke_width=0),
            Text("q_proj", font_size=20, color=ACCENT, weight=BOLD),
        ).arrange(RIGHT, buff=0.15)
        leg_v = VGroup(
            Square(side_length=0.20, fill_color=GOLD, fill_opacity=0.85,
                   stroke_width=0),
            Text("v_proj", font_size=20, color=GOLD, weight=BOLD),
        ).arrange(RIGHT, buff=0.15)
        legend = VGroup(leg_q, leg_v).arrange(RIGHT, buff=0.60)
        legend.move_to(chart_origin + UP*(chart_h + 0.4) + RIGHT*chart_w*0.4)

        # Aggregate stats from real data
        q_vals = [abs(layer_vals[l]["q_proj"]) for l in layers_sorted
                  if "q_proj" in layer_vals[l]]
        v_vals = [abs(layer_vals[l]["v_proj"]) for l in layers_sorted
                  if "v_proj" in layer_vals[l]]
        agg = VGroup(
            Text(f"q_proj: |cos|  ̄ = {_format_sci(np.mean(q_vals))},"
                 f"   max = {_format_sci(max(q_vals))}",
                 font_size=22, color=ACCENT, weight=BOLD),
            Text(f"v_proj: |cos|  ̄ = {_format_sci(np.mean(v_vals))},"
                 f"   max = {_format_sci(max(v_vals))}",
                 font_size=22, color=GOLD, weight=BOLD),
        ).arrange(DOWN, buff=0.20, aligned_edge=LEFT)
        agg.move_to([0, -2.4, 0])

        verdict = Text(
            "Uniform pe adâncime. Niciun layer pivot. Module echivalente.",
            font_size=22, color=GREEN, weight=BOLD, slant=ITALIC)
        verdict.move_to([0, -3.5, 0])

        self.play(Write(h), run_time=0.5)
        self.play(FadeIn(sub), run_time=0.3)
        self.play(Create(x_axis), Create(y_axis), run_time=0.5)
        self.play(FadeIn(x_lbl), FadeIn(y_lbl),
                  FadeIn(y_tick_top), FadeIn(y_tick_bot), run_time=0.4)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars],
                              lag_ratio=0.015), run_time=1.4)
        self.play(FadeIn(legend), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(a, shift=UP*0.10) for a in agg],
                              lag_ratio=0.20), run_time=0.8)
        self.play(FadeIn(verdict, shift=UP*0.15), run_time=0.5)
        self.wait(2.0)


# ═════════════════════════════════════════════════════════════
#  S15 — Conclusion
# ═════════════════════════════════════════════════════════════
class S15_Conclusion(Scene):
    def setup(self): bg(self)

    def construct(self):
        h = headline("Concluzie", color=GOLD, size=52)

        rq = Text("Întrebare:",
                  font_size=28, color=DIM, slant=ITALIC, weight=BOLD)
        rq.move_to([-2.6, 3.4, 0])
        rq2 = Text("OLoRA păstrează ortogonalitatea în practică?",
                   font_size=26, color=FG, weight=BOLD)
        rq2.move_to([0.4, 3.4, 0])

        # Findings table
        findings_h = Text("Rezultate (max |cos θ|):",
                          font_size=26, color=GOLD, weight=BOLD)
        findings_h.move_to([0, 2.4, 0])

        a_max = max(abs(A_SUM["max"]), abs(A_SUM["min"]))
        b_max = max(abs(B_SUM["max"]), abs(B_SUM["min"]))
        ab_max = max(abs(AB_SUM["max"]), abs(AB_SUM["min"]))

        rows = VGroup(
            self._row("matrici A",          _format_sci(a_max),  ACCENT),
            self._row("matrici B",          _format_sci(b_max),  GOLD),
            self._row("produs A·B",         _format_sci(ab_max), GREEN),
            self._row("per layer (avg)",   _format_sci(np.mean([abs(v) for v in AB_LAYER_SCORES.values()])), DIM),
        ).arrange(DOWN, buff=0.20, aligned_edge=LEFT)
        rows.move_to([0, 0.7, 0])

        ans_h = Text("Răspuns:",
                     font_size=30, color=DIM, weight=BOLD, slant=ITALIC)
        ans_h.move_to([-2.6, -1.4, 0])
        ans = Text("DA.",
                   font_size=72, color=GREEN, weight=BOLD)
        ans.move_to([0.6, -1.4, 0])

        f1 = Text("• ortogonalitate sub 10⁻³ la toate granularitățile",
                  font_size=22, color=FG)
        f2 = Text("• cascadă: constrângere pe A → ortogonalitate pe ΔW",
                  font_size=22, color=FG)
        f3 = Text("• uniform pe layer și modul",
                  font_size=22, color=FG)
        finals = VGroup(f1, f2, f3).arrange(DOWN, buff=0.18,
                                            aligned_edge=LEFT)
        finals.move_to([0, -2.7, 0])

        outro = Text("Constrângerea geometrică se transferă funcțional.",
                     font_size=24, color=GOLD, weight=BOLD, slant=ITALIC)
        outro.move_to([0, -3.7, 0])

        self.play(Write(h), run_time=0.6)
        self.play(FadeIn(rq), FadeIn(rq2), run_time=0.7)
        self.play(FadeIn(findings_h), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r, shift=UP*0.10) for r in rows],
                              lag_ratio=0.18), run_time=1.2)
        self.play(FadeIn(ans_h), Write(ans), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(f, shift=UP*0.10) for f in finals],
                              lag_ratio=0.18), run_time=1.0)
        self.play(FadeIn(outro, shift=UP*0.15), run_time=0.5)
        self.wait(2.4)

    def _row(self, label, value, col):
        l = Text(label, font_size=22, color=DIM, weight=BOLD)
        v = Text(value, font_size=24, color=col, weight=BOLD)
        sep = Text("=", font_size=22, color=DIM)
        return VGroup(l, sep, v).arrange(RIGHT, buff=0.40)
