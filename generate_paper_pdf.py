"""
Generate Peer-Reviewed IEEE (ieeeconf) Preprint PDF
Title: Tactile Contact-Manifold Safety Filtering for Learned Manipulation Policies:
       A 1 kHz Multi-Rate Control Barrier Function Architecture for Anti-Jamming and Contact Passivity
Authors: Conor O'Neill* and Alec Vysotski* (*Equal contribution)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable, Image
)
from reportlab.pdfgen import canvas


def render_latex_equation_box(
    latex_lines: list,
    eq_num: str,
    out_path: str,
    width_in: float = 7.05,
    height_in: float = 0.52,
    fontsize: float = 9.2
) -> str:
    """Renders compiled LaTeX math lines into a crisp 600-DPI IEEE equation block image."""
    plt.rcParams["mathtext.fontset"] = "cm"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig = plt.figure(figsize=(width_in, height_in), dpi=600)
    fig.patch.set_facecolor("#FFFFFF")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor("#FFFFFF")
    for spine in ax.spines.values():
        spine.set_edgecolor("#E2E8F0")
        spine.set_linewidth(0.6)
    ax.set_xticks([])
    ax.set_yticks([])

    n = len(latex_lines)
    for idx, line in enumerate(latex_lines):
        y_pos = 1.0 - (idx + 0.5) / n
        ax.text(
            0.47,
            y_pos,
            line,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#0F172A",
            transform=ax.transAxes,
        )

    if eq_num:
        ax.text(
            0.985,
            0.5,
            eq_num,
            ha="right",
            va="center",
            fontsize=fontsize,
            fontfamily="serif",
            color="#0F172A",
            transform=ax.transAxes,
        )

    plt.savefig(out_path, dpi=600, facecolor=fig.get_facecolor())
    plt.close(fig)
    return os.path.abspath(out_path)


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Times-Roman", 8)
        self.setFillColor(colors.HexColor("#334155"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(48, 11 * inch - 32, "TACTILE CONTACT-MANIFOLD SAFETY FILTERING FOR LEARNED MANIPULATION POLICIES")
            self.drawRightString(8.5 * inch - 48, 11 * inch - 32, "O'NEILL AND VYSOTSKI (IEEECONF PREPRINT, 2026)")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(48, 11 * inch - 36, 8.5 * inch - 48, 11 * inch - 36)

        # Footer
        page_str = f"{self._pageNumber}"
        self.drawCentredString(4.25 * inch, 30, page_str)
        self.drawString(48, 30, "IEEE Conference Format (ieeeconf) — Robotics (cs.RO) / Systems and Control (eess.SY)")
        self.drawRightString(8.5 * inch - 48, 30, f"Page {self._pageNumber} of {total_pages}")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(48, 40, 8.5 * inch - 48, 40)
        self.restoreState()


def create_paper_pdf(output_path: str = "docs/research_paper.pdf"):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    eq_dir = os.path.join("docs", "figures", "equations")
    os.makedirs(eq_dir, exist_ok=True)

    # Pre-render 600 DPI LaTeX equation blocks
    eq1_path = render_latex_equation_box(
        [
            r"$\Delta \mathbf{r}_c = \frac{1}{2}\left(\mathbf{r}_{c,L} - \mathbf{r}_{c,R}\right) \in \mathbb{R}^2, \qquad "
            r"\mathbf{C}_p = \int_{\Omega} \left(\mathbf{u} - \mathbf{r}_{c,p}\right)\left(\mathbf{u} - \mathbf{r}_{c,p}\right)^\top f_{z,p}(u,v)\,\mathrm{d}u\,\mathrm{d}v \succeq \mathbf{0}$"
        ],
        "(1)",
        os.path.join(eq_dir, "eq1_invariants.png"),
        height_in=0.38,
        fontsize=9.4,
    )

    eq2_path = render_latex_equation_box(
        [
            r"$\hat{\mathbf{F}}_{\mathrm{ext}}(t) = (1 - \gamma_F)\mathbf{F}_{\mathrm{prop}}(t) + \gamma_F \mathbf{F}_{\mathrm{tac}}(t - \tau_{\mathrm{delay}}), \quad "
            r"\hat{\boldsymbol{\theta}}_{\mathrm{est}}(t) = \boldsymbol{\theta}_{\mathrm{wrist}}(t) + \gamma_{\mathrm{obs}}\,\boldsymbol{\delta}(t)$"
        ],
        "(2)",
        os.path.join(eq_dir, "eq2_observer.png"),
        height_in=0.38,
        fontsize=9.0,
    )

    eq3_path = render_latex_equation_box(
        [
            r"$\rho_{\mathrm{stick}} = \frac{\int_{\Omega} \mathbb{I}\left(\|\mathbf{f}_{xy}(u,v)\|_2 < \mu f_z(u,v)\right)\mathrm{d}u\,\mathrm{d}v}{\int_{\Omega} \mathbb{I}\left(f_z(u,v) > 0\right)\mathrm{d}u\,\mathrm{d}v} \in [0,1], \qquad "
            r"J_{\mathrm{jam}} = \frac{|\hat{F}_{z,\mathrm{ext}}|\,r_{\mathrm{char}}}{\|\hat{\boldsymbol{\tau}}_{\mathrm{tilt}}\|_2 + \epsilon}$"
        ],
        "(3)",
        os.path.join(eq_dir, "eq3_stick_jam.png"),
        height_in=0.42,
        fontsize=9.1,
    )

    eq4_path = render_latex_equation_box(
        [
            r"$\kappa(\mathbf{C}) = \mathrm{clip}\left(\frac{\mathrm{tr}(\mathbf{C})}{C_{\mathrm{ref}}},\, \kappa_{\min},\, 1.05\right), \quad "
            r"\mathbf{K}_{\mathrm{rot}}(\mathbf{C}) = K_{\mathrm{rot},0}\,\kappa(\mathbf{C})\left(\eta_0\mathbf{I}_2 + (1-\eta_0)\frac{\mathbf{C}}{\mathrm{tr}(\mathbf{C})+\epsilon}\right), \quad "
            r"\boldsymbol{\tau}_{\mathrm{align}} = -\mathbf{K}_{\mathrm{rot}}(\mathbf{C})\hat{\boldsymbol{\theta}}_{\mathrm{est}}(t) + K_{\tau}\hat{\boldsymbol{\tau}}_{\mathrm{ext}}(t)$"
        ],
        "(4)",
        os.path.join(eq_dir, "eq4_tensor_control.png"),
        height_in=0.40,
        fontsize=8.5,
    )

    eq5a_path = render_latex_equation_box(
        [
            r"$\min_{\mathbf{u}\in\mathbb{R}^6,\,\delta\in\mathbb{R}} \; \frac{1}{2}\left\|\mathbf{u} - \mathbf{u}_{\mathrm{target}}\right\|_{\mathbf{W}}^2 + \frac{1}{2}\lambda_{\mathrm{slack}}\delta^2 \quad \mathrm{s.t.} \quad u_z + \delta \geq u_{z,\mathrm{cbf}}(h_{\mathrm{force}}, h_{\mathrm{jam}}), \quad |u_x|,\,|u_y| \leq F_{\mathrm{lat},\max}(h_{\mathrm{slip}}),$",
            r"$E_{k+1} = E_k + \Delta t\left(0.8 P_{\mathrm{diss}} - \mathbf{u}_{\mathrm{net}}^\top \mathbf{v}_{\mathrm{ee}}\right) \geq E_{\min}$"
        ],
        "(5a)",
        os.path.join(eq_dir, "eq5a_cbf_qp.png"),
        height_in=0.55,
        fontsize=8.8,
    )

    eq5b_path = render_latex_equation_box(
        [
            r"$u_{z,\mathrm{cbf}}(h_{\mathrm{force}}, h_{\mathrm{jam}}) = \left\{ \;\; \mathrm{Stage\,1\;(Free\text{-}Space):}\;\; -F_{\mathrm{app}} - D_1 \dot{z}_{\mathrm{ee}}, \qquad "
            r"\mathrm{Stage\,2\;(Un\text{-}Wedging\;at\;}G_{1\rightarrow 2}\mathrm{):}\;\; K_z(z_{\mathrm{clear}} - z_{\mathrm{ee}}) - D_2 \dot{z}_{\mathrm{ee}} + \beta_f \|\hat{\mathbf{F}}_{\mathrm{ext}}\|, \right.$",
            r"$\left. \mathrm{Stage\,3\;(Aligned\;Insertion\;at\;}G_{2\rightarrow 3}\mathrm{):}\;\; -\min\left(F_{z,\mathrm{nom}},\, \frac{\alpha_f}{K_{\mathrm{loc}}}h_{\mathrm{force}}\right) - D_3 \dot{z}_{\mathrm{ee}} \;\; \right\}$"
        ],
        "(5b)",
        os.path.join(eq_dir, "eq5b_piecewise_uz.png"),
        height_in=0.56,
        fontsize=8.5,
    )

    eq6_path = render_latex_equation_box(
        [
            r"$F_{\mathrm{grip}}(t) = F_{\mathrm{grip},\min}(h_{\mathrm{slip}}) = F_{\mathrm{grip},0} + K_{\mathrm{slip}}\max\left(0,\, \rho_{\mathrm{safe}} - \hat{\rho}_{\mathrm{stick}}(t)\right) + \mu_{\mathrm{safe}}^{-1}\left\|\hat{\mathbf{F}}_{xy,\mathrm{ext}}(t)\right\|_2 \quad \mathrm{(1D\;Closed\text{-}Form\;Gripper\;Barrier)}$"
        ],
        "(6)",
        os.path.join(eq_dir, "eq6_grip_cbf.png"),
        height_in=0.36,
        fontsize=8.9,
    )

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=48,
        rightMargin=48,
        topMargin=44,
        bottomMargin=44
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'PaperTitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=15.2,
        leading=18.5,
        alignment=1,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=3
    )

    subtitle_style = ParagraphStyle(
        'PaperSubtitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=10.5,
        leading=13.2,
        alignment=1,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=6
    )

    authors_style = ParagraphStyle(
        'PaperAuthors',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=10.5,
        leading=13.0,
        alignment=1,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=2
    )

    affil_style = ParagraphStyle(
        'PaperAffil',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=8.6,
        leading=11.0,
        alignment=1,
        textColor=colors.HexColor('#334155'),
        spaceAfter=7
    )

    abstract_body = ParagraphStyle(
        'AbstractBody',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=8.3,
        leading=10.6,
        alignment=4,
        textColor=colors.HexColor('#0F172A')
    )

    sec_heading = ParagraphStyle(
        'SecHeading',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=9.8,
        leading=12.4,
        alignment=1,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=6,
        spaceAfter=2.5,
        keepWithNext=True
    )

    subsec_heading = ParagraphStyle(
        'SubSecHeading',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=8.9,
        leading=11.2,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=3.5,
        spaceAfter=1.8,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8.8,
        leading=11.0,
        alignment=4,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=3.0
    )

    caption_style = ParagraphStyle(
        'FigCaption',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=7.8,
        leading=9.8,
        alignment=4,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=2,
        spaceAfter=5
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=7.5,
        leading=9.2,
        alignment=1,
        textColor=colors.HexColor('#0F172A')
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=7.5,
        leading=9.2,
        alignment=1,
        textColor=colors.white
    )

    story = []

    # ---------------------------------------------------------
    # TITLE & HEADER (IEEECONF STYLE)
    # ---------------------------------------------------------
    story.append(Paragraph("Tactile Contact-Manifold Safety Filtering for Learned Manipulation Policies:", title_style))
    story.append(Paragraph("A 1 kHz Multi-Rate Control Barrier Function Architecture for Anti-Jamming and Contact Passivity", subtitle_style))
    story.append(Paragraph("Conor O'Neill<sup>*</sup> and Alec Vysotski<sup>*</sup>", authors_style))
    story.append(Paragraph("<sup>*</sup>Equal contribution &nbsp;&mdash;&nbsp; Independent Research Preprint (IEEE <i>ieeeconf</i> Format)", affil_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#94A3B8"), spaceAfter=4))

    # ---------------------------------------------------------
    # ABSTRACT
    # ---------------------------------------------------------
    abstract_text = (
        "<i>Abstract</i>&mdash;Learned manipulation policies such as action-chunking diffusion policies generalize well visually but can "
        "produce large force spikes, two-point wedging, and object drop when contact geometry is out of distribution (OOD). We present a "
        "1 kHz supervisory safety filter that sits between a pre-trained policy and the robot actuators and requires no policy retraining. "
        "A complementary observer fuses 1 kHz proprioception with delayed (10&ndash;50 ms) optical-tactile traction invariants&mdash;the "
        "Differential Center of Pressure &Delta;<b>r</b><sub><i>c</i></sub>, the Contact Constraint Tensor <b>C</b>, the Incipient Stick Ratio "
        "&rho;<sub>stick</sub>, and a Whitney-style Jamming Index <i>J</i><sub>jam</sub>. A 7-variable OSQP quadratic program filters the 6-DoF arm "
        "wrench (mean 0.11 ms per filter step in Python), a closed-form 1D barrier regulates grip force, a 3-stage hybrid automaton performs "
        "un-wedging before reorientation, and a discrete-time energy tank bounds energy injected by delayed feedback. We evaluate in MuJoCo "
        "with an analytical tactile model and a stochastic action-chunking policy proxy. On 0.15 mm-clearance peg insertion with <i>N</i> = 50 "
        "held-out random OOD initial conditions (per-axis tilt up to &plusmn;15&deg;, lateral offset up to &plusmn;0.92 mm), the filter achieves "
        "74% insertion success (95% CI 60&ndash;84%; 97% inside an a-priori capture basin) with zero jams and all trials below the 15 N force limit, "
        "versus 8% success and 92% jamming for 6-axis wrist F/T admittance and 0% success for the unconstrained policy. On stiff contour tracking "
        "(<i>N</i> = 20) it achieves 100% safe tracking (2.06 N RMSE), and under incipient slip (<i>N</i> = 20) it reduces tool drop from "
        "85&ndash;100% to 10%. All results are simulation-only; we discuss the idealizations of the tactile model and the hand-tuned components explicitly."
    )

    abs_table = Table(
        [[Paragraph(abstract_text, abstract_body)]],
        colWidths=[7.15 * inch]
    )
    abs_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(abs_table)
    story.append(Spacer(1, 3))

    # ---------------------------------------------------------
    # SECTION I: INTRODUCTION
    # ---------------------------------------------------------
    story.append(Paragraph("I. INTRODUCTION", sec_heading))
    story.append(Paragraph(
        "Visuomotor imitation learning and action-chunking diffusion policies [7], [8] let robots acquire manipulation skills from demonstrations. "
        "Tactile-conditioned variants such as FARM [1] additionally consume 3D force-distribution estimates from optical tactile sensors (FEATS [2]). "
        "These policies nonetheless provide no physical safety guarantee. When a grasped part meets an OOD constraint&mdash;for example, an angular "
        "misalignment of several degrees during sub-millimeter-clearance insertion&mdash;the policy may extrapolate and drive the part into the receptacle. "
        "In stiff environments (<i>K</i><sub>env</sub> &ge; 10<sup>4</sup> N/m) this produces force spikes above 40 N within tens of milliseconds and two-point wedging.",
        body_style
    ))
    story.append(Paragraph(
        "Classical compliance does not resolve this trade-off. Low-gain isotropic impedance limits impact force but stalls on the rim because it lacks directional "
        "authority to reorient the part. Wrist F/T admittance suffers from single-point wrench ambiguity: it cannot separate a tip-shear moment "
        "(<b>r</b><sub>tip</sub> &times; <b>F</b><sub>shear</sub>) from a pure alignment couple, nor observe incipient slip inside the grasp.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Contributions:</b> (1) A tactile-invariant extractor and multi-rate observer that fuses 25&ndash;50 Hz dual-pad traction fields with 1 kHz "
        "proprioception, gated by a contact-latching threshold. (2) A supervisory filter combining a 7-variable CBF-QP, a closed-form grip barrier, a 3-stage "
        "anti-jamming automaton, and an energy-tank passivity supervisor. (3) A simulation study with held-out evaluation, three baselines, component ablations, "
        "and robustness sweeps, together with an explicit account of which components are hand-tuned and how the tactile signal is idealized.",
        body_style
    ))

    # ---------------------------------------------------------
    # SECTION II: RELATED WORK
    # ---------------------------------------------------------
    story.append(Paragraph("II. RELATED WORK", sec_heading))
    story.append(Paragraph(
        "<b>Optical Tactile Force Estimation:</b> Vision-based elastomeric sensors (GelSight Mini, DIGIT) capture high-resolution surface deformation. "
        "FEATS [2] maps GelSight images to 3D traction distributions <b>f</b>(<i>u</i>,<i>v</i>) in real time using FEA-calibrated learning, and TacEx [3] "
        "provides GPU-accelerated elastomer simulation. Whereas FARM [1] feeds high-dimensional tactile observations to a neural policy, we project "
        "<b>f</b>(<i>u</i>,<i>v</i>) onto low-dimensional mechanical invariants usable inside a 1 kHz convex filter.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Control Barrier Functions and Passivity:</b> Control Barrier Functions [4] enforce forward invariance of safe sets via quadratic programming. "
        "Energy tanks [5] bound energy injection under variable impedance and delayed feedback. Our filter combines these tools with Cattaneo-Mindlin partial-slip "
        "reasoning and Whitney's quasi-static assembly analysis [6].",
        body_style
    ))

    # ---------------------------------------------------------
    # SECTION III: CONTACT MECHANICS & MULTI-RATE TACTILE OBSERVER
    # ---------------------------------------------------------
    story.append(Paragraph("III. CONTACT MECHANICS AND MULTI-RATE TACTILE INVARIANTS", sec_heading))
    story.append(Paragraph(
        "Consider a parallel-jaw gripper with left (<i>L</i>) and right (<i>R</i>) GelSight-type pads (domain &Omega;, 32 &times; 32 grid) grasping a "
        "10 &times; 10 &times; 50 mm workpiece. Each pad <i>p</i> &isin; {<i>L</i>, <i>R</i>} measures a traction field "
        "<b>f</b><sub><i>p</i></sub>(<i>u</i>,<i>v</i>) = [<i>f<sub>x,p</sub></i>, <i>f<sub>y,p</sub></i>, <i>f<sub>z,p</sub></i>]<sup><i>T</i></sup>.",
        body_style
    ))
    story.append(Paragraph("A. Differential Center of Pressure and Contact Constraint Tensor", subsec_heading))
    story.append(Paragraph(
        "With <b>r</b><sub><i>c,p</i></sub> = <i>F</i><sub><i>z,p</i></sub><sup>&minus;1</sup> &int;<sub>&Omega;</sub> <b>u</b> <i>f<sub>z,p</sub></i> d<i>u</i> d<i>v</i> "
        "the normal CoP on pad <i>p</i>:",
        body_style
    ))
    story.append(Image(eq1_path, width=7.05 * inch, height=0.38 * inch))
    story.append(Spacer(1, 2))
    story.append(Paragraph("B. Multi-Rate Complementary Observer and Contact Latching", subsec_heading))
    story.append(Paragraph(
        "Optical tactile estimation runs at 25&ndash;50 Hz (&tau;<sub>delay</sub> &isin; [10, 50] ms), too slow alone to arrest a stiff-contact impact; "
        "proprioception runs at 1 kHz but is wrench-ambiguous. We fuse them (&gamma;<sub><i>F</i></sub> = 0.30, &gamma;<sub>obs</sub> = 0.25):",
        body_style
    ))
    story.append(Image(eq2_path, width=7.05 * inch, height=0.38 * inch))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "where the latched tactile residual is <b>&delta;</b> = arcsin(&Delta;<b>r</b><sub><i>c</i></sub>(<i>t</i> &minus; &tau;)/<i>r</i><sub>pad</sub>) &minus; "
        "<b>&theta;</b><sub>wrist</sub>(<i>t</i> &minus; &tau;) when the delayed contact force satisfies ||<b>F</b>(<i>t</i> &minus; &tau;)|| &ge; 1.0 N, "
        "and otherwise decays as <b>&delta;</b> &larr; 0.90 <b>&delta;</b> per 1 ms step. The latching threshold is intended to reject low-force marker noise "
        "and viscoelastic relaxation during unloading on real sensors; this has not yet been validated on hardware.",
        body_style
    ))
    story.append(Paragraph("C. Incipient Stick Ratio and Jamming Index", subsec_heading))
    story.append(Image(eq3_path, width=7.05 * inch, height=0.42 * inch))

    # ---------------------------------------------------------
    # SECTION IV: 1 kHz SUPERVISORY CBF-QP & PASSIVITY ARCHITECTURE
    # ---------------------------------------------------------
    story.append(Paragraph("IV. SUPERVISORY FILTER", sec_heading))
    story.append(Paragraph("A. Tensor Gain Modulation", subsec_heading))
    story.append(Paragraph(
        "With <b>C</b> = &frac12;(<b>C</b><sub><i>L</i></sub> + <b>C</b><sub><i>R</i></sub>), we define the directional alignment gain:",
        body_style
    ))
    story.append(Image(eq4_path, width=7.05 * inch, height=0.40 * inch))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "with &kappa;<sub>min</sub> = 0.92, <i>C</i><sub>ref</sub> = 10<sup>&minus;4</sup>, &eta;<sub>0</sub> = 0.85, and "
        "<b>&tau;</b><sub>align</sub> = &minus;<b>K</b><sub>rot</sub><b>&theta;</b><sub>est</sub> + <i>K</i><sub>&tau;</sub><b>&tau;</b><sub>ext</sub>. "
        "Note that with &eta;<sub>0</sub> = 0.85 and &kappa; &isin; [0.92, 1.05] the modulation is mild; ablation 1 (Table III) removes "
        "&Delta;<b>r</b><sub><i>c</i></sub> and <b>C</b> jointly, so we do not claim a separate contribution of <b>C</b>.",
        body_style
    ))
    story.append(Paragraph("B. 7-Variable CBF-QP and Hybrid Automaton", subsec_heading))
    story.append(Paragraph(
        "Given the 20 Hz policy wrench <b>u</b><sub>&pi;</sub> (chunk horizon <i>H</i> = 50), at 1 kHz we solve over <b>z</b> = [<b>u</b><sup><i>T</i></sup>, &delta;]<sup><i>T</i></sup> &isin; <b>R</b><sup>7</sup>:",
        body_style
    ))
    story.append(Image(eq5a_path, width=7.05 * inch, height=0.55 * inch))
    story.append(Spacer(1, 1.5))
    story.append(Image(eq5b_path, width=7.05 * inch, height=0.56 * inch))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "with guards <i>G</i><sub>1&rarr;2</sub>: ||<b>F</b>|| &gt; 0.1 N &and; (<i>h</i><sub>jam</sub> &lt; 0 &or; ||<b>&theta;</b>|| &ge; 0.65&deg;) and "
        "<i>G</i><sub>2&rarr;3</sub>: <i>z</i> &gt; &minus;3.2 mm after the tactile frame has arrived.<br/>"
        "<b>Hand-tuned rim-arrest guard (disclosure):</b> Stage 3 contains a heuristic for an aligned part that lands on the flat rim outside the capture region: "
        "within a height band above the rim the axial damping is raised to 52 Ns/m; if vertical contact force exceeds 4.2 N the Stage 2 retraction law is applied, "
        "and above 6.5 N it is latched. These constants were selected while observing the structured scenario set (Sec. V); performance is therefore reported "
        "primarily on held-out random trials.",
        body_style
    ))
    story.append(Paragraph("C. Decoupled Grip Barrier", subsec_heading))
    story.append(Image(eq6_path, width=7.05 * inch, height=0.36 * inch))
    story.append(Spacer(1, 2))
    story.append(Paragraph("D. Safety and Passivity Arguments", subsec_heading))
    story.append(Paragraph(
        "<b>Proposition 1 (informal):</b> Under quasi-static contact with stiffness <i>K</i><sub>env</sub> and the automaton (Eqs. 5a&ndash;5b), if Stage 2 "
        "is entered with <i>h</i><sub>force</sub> &gt; 0 and the Stage 2 bound produces <i>v<sub>z</sub></i> &gt; 0 before ||<b>F</b>|| reaches "
        "<i>F</i><sub>max</sub> = 15 N, then <i>h</i><sub>force</sub> remains non-negative through the transition to Stage 3, which begins only after rim "
        "compression is relieved.<br/>"
        "<i>Sketch:</i> At <i>G</i><sub>1&rarr;2</sub>, <i>h</i><sub>force</sub> &approx; 14.9 N. Stage 2 commands a restoring wrench that reverses "
        "<i>v<sub>z</sub></i> after peak compression &Delta;<i>z</i><sub>max</sub> &approx; <i>v</i><sub>0</sub>(<i>m</i>/<i>K</i><sub>env</sub>)<sup>1/2</sup>, "
        "after which d<i>h</i><sub>force</sub>/d<i>t</i> = <i>K</i><sub>env</sub><i>v<sub>z</sub></i> &gt; 0. This is a sufficient-condition argument, "
        "not a formal hybrid-systems proof; the condition depends on approach velocity and is verified empirically below.<br/>"
        "<b>Proposition 2 (passivity):</b> With storage <i>V</i> = &frac12;<b>v</b><sup><i>T</i></sup><b>M</b><b>v</b> + <i>E<sub>k</sub></i> and "
        "<i>E<sub>k</sub></i> &ge; <i>E</i><sub>min</sub> = 0.05 J, attenuating delayed tactile feedback whenever it would inject positive power bounds the "
        "energy injected by delayed updates by <i>E</i><sub>0</sub> &minus; <i>E</i><sub>min</sub>.",
        body_style
    ))

    # ---------------------------------------------------------
    # SECTION V: EXPERIMENTS
    # ---------------------------------------------------------
    story.append(Paragraph("V. EXPERIMENTS", sec_heading))
    story.append(Paragraph(
        "<b>Simulation Setup and Idealizations:</b> All experiments run in MuJoCo 3.15 at 1 kHz. The end-effector is modeled as a 6-DoF operational-space "
        "carriage coupled to a compliant in-hand peg mount (<i>K</i><sub>grip</sub> = 1500 N/m, <i>K</i><sub>rot,grip</sub> = 8.0 N&middot;m/rad), kinematically "
        "synchronized to a 7-DoF Panda model. The nominal policy is a 20 Hz <i>StochasticActionChunkingPolicy</i> (<i>H</i> = 50) with OOD extrapolation bias and "
        "Ornstein-Uhlenbeck noise; it is a <i>proxy</i> for FARM, not the trained FARM model.<br/>"
        "<b>Tactile idealization:</b> Pad traction fields are synthesized analytically (Cattaneo-Mindlin patch model). During contact (||<b>F</b>|| &ge; 0.1 N) "
        "the pad CoP offsets are computed from the true peg orientation plus in-hand deflection, without sensor noise, hysteresis, or learned-estimator error. "
        "The tactile channel therefore approximates a noise-free tilt measurement, and all tactile-dependent results should be read as an upper bound pending "
        "hardware validation with FEATS-style estimation.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Benchmarks:</b> <i>Benchmark A (insertion, 0.15 mm clearance per side):</i> <b>Held-out set (primary):</b> <i>N</i> = 50 initial conditions sampled "
        "once with an unused seed, &theta;<sub><i>x</i>,0</sub>, &theta;<sub><i>y</i>,0</sub> &sim; U(&minus;15&deg;, 15&deg;) with ||<b>&theta;</b><sub>0</sub>||<sub>2</sub> "
        "&le; 19.2&deg;, and lateral offsets &Delta;<i>x</i><sub>0</sub>, &Delta;<i>y</i><sub>0</sub> &sim; U(&minus;0.92, 0.92) mm. The capture basin is defined "
        "a priori as ||<b>&theta;</b><sub>0</sub>||<sub>2</sub> &le; 15&deg; and max(|&Delta;<i>x</i><sub>0</sub>|, |&Delta;<i>y</i><sub>0</sub>|) &le; 0.80 mm "
        "(32/50 held-out trials). <b>Structured stress set:</b> 50 hand-designed conditions (47 in-basin, 3 extreme compound cases) used during development, "
        "including for tuning the rim-arrest guard.<br/>"
        "<i>Benchmark B (contour tracking):</i> <i>N</i> = 20 phase-shifted stiff sinusoidal contours, <i>K</i><sub>env</sub> = 10<sup>4</sup> N/m, <i>v<sub>x</sub></i> = 0.05 m/s, &tau;<sub>delay</sub> = 25 ms.<br/>"
        "<i>Benchmark C (incipient slip):</i> <i>N</i> = 20 manipulation trials of a 0.45 kg tool under shear pulses, &mu; &isin; [0.22, 0.58].",
        body_style
    ))

    # TABLE I: Held-out Primary + Structured Stress Set
    story.append(Paragraph("TABLE I: Benchmark A: 0.15 mm-Clearance Insertion. Held-out random set is the primary result; structured set was used during development. CIs are Wilson 95%.", subsec_heading))
    table1_data = [
        [Paragraph("Controller / Policy", table_header),
         Paragraph("Held-Out Succ. (N=50)", table_header),
         Paragraph("Held-Out In-Basin", table_header),
         Paragraph("Held-Out Jam / Peak", table_header),
         Paragraph("Structured Succ. (N=50)", table_header),
         Paragraph("Structured Jam / Peak", table_header)],
        [Paragraph("Unconstrained Policy (Proxy)", table_cell),
         Paragraph("0.0% (0/50) [0&ndash;7%]", table_cell),
         Paragraph("0/32 (0.0%)", table_cell),
         Paragraph("100% | 42.96 &plusmn; 2.61 N", table_cell),
         Paragraph("6.0% (3/50)", table_cell),
         Paragraph("94.0% | 41.87 &plusmn; 7.43 N", table_cell)],
        [Paragraph("Wrist F/T Admittance", table_cell),
         Paragraph("8.0% (4/50) [3&ndash;19%]", table_cell),
         Paragraph("3/32 (9.4%)", table_cell),
         Paragraph("92.0% | 33.54 &plusmn; 3.82 N", table_cell),
         Paragraph("20.0% (10/50)", table_cell),
         Paragraph("80.0% | 32.80 &plusmn; 6.00 N", table_cell)],
        [Paragraph("Fixed Impedance (K=200)", table_cell),
         Paragraph("0.0% (Rim Stall)", table_cell),
         Paragraph("0/32 (0.0%)", table_cell),
         Paragraph("0.0% | 9.99 &plusmn; 0.17 N", table_cell),
         Paragraph("0.0% (Rim Stall)", table_cell),
         Paragraph("0.0% | 10.31 &plusmn; 0.21 N", table_cell)],
        [Paragraph("<b>Tactile Safety Filter (Ours)</b>", table_cell),
         Paragraph("<b>74.0% (37/50) [60&ndash;84%]</b>", table_cell),
         Paragraph("<b>31/32 (96.9%)</b>", table_cell),
         Paragraph("<b>0.0% | 10.15 &plusmn; 2.29 N</b>", table_cell),
         Paragraph("<b>94.0% (47/50)</b>", table_cell),
         Paragraph("<b>0.0% | 8.83 &plusmn; 2.58 N</b>", table_cell)]
    ]
    t1 = Table(table1_data, colWidths=[1.85 * inch, 1.25 * inch, 0.95 * inch, 1.15 * inch, 1.05 * inch, 1.05 * inch])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#ECFDF5')),
        ('TOPPADDING', (0, 0), (-1, -1), 2.8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.8),
    ]))
    story.append(t1)
    story.append(Spacer(1, 3.0))

    # TABLE II & III SIDE-BY-SIDE OR SEQUENTIAL
    story.append(Paragraph("TABLE II: Benchmarks B &amp; C (<i>N</i> = 20 Trials Each) &mdash; Non-Planar Contour Tracking &amp; Incipient-Slip Tool Manipulation", subsec_heading))
    table2_bc_data = [
        [Paragraph("Controller Architecture", table_header),
         Paragraph("Bench B: Safe Track (%)", table_header),
         Paragraph("Bench B: Force RMSE", table_header),
         Paragraph("Bench C: Drop Rate (%)", table_header),
         Paragraph("Bench C: Mean Slip / Grip", table_header)],
        [Paragraph("Unconstrained Policy (Proxy)", table_cell),
         Paragraph("15.0% (3/20)", table_cell),
         Paragraph("6.80 N (Peak 20.37 N)", table_cell),
         Paragraph("100.0% (0% Succ.)", table_cell),
         Paragraph("148.8 mm / 6.20 N", table_cell)],
        [Paragraph("Fixed Impedance (K=200)", table_cell),
         Paragraph("85.0% (17/20)", table_cell),
         Paragraph("2.53 N (Peak 12.43 N)", table_cell),
         Paragraph("85.0% (15% Succ.)", table_cell),
         Paragraph("104.8 mm / 6.80 N", table_cell)],
        [Paragraph("Wrist F/T Admittance", table_cell),
         Paragraph("0.0% (Chatter)", table_cell),
         Paragraph("27.38 N (Peak 127.59 N)", table_cell),
         Paragraph("85.0% (15% Succ.)", table_cell),
         Paragraph("44.0 mm / 13.25 N", table_cell)],
        [Paragraph("<b>Tactile Safety Filter (Ours)</b>", table_cell),
         Paragraph("<b>100.0% (20/20 &le; 15 N)</b>", table_cell),
         Paragraph("<b>2.06 N (Peak 10.64 N)</b>", table_cell),
         Paragraph("<b>10.0% (90% Succ.)</b>", table_cell),
         Paragraph("<b>0.44 mm / 14.60 N</b>", table_cell)]
    ]
    t2_bc = Table(table2_bc_data, colWidths=[2.15 * inch, 1.25 * inch, 1.25 * inch, 1.15 * inch, 1.35 * inch])
    t2_bc.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#ECFDF5')),
        ('TOPPADDING', (0, 0), (-1, -1), 2.8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.8),
    ]))
    story.append(t2_bc)
    story.append(Spacer(1, 3.0))

    story.append(Paragraph("TABLE III: Component Ablations (<i>N</i> = 20) &amp; Multi-Seed Environmental Sweeps (<i>N</i> = 5 Seeds/Point)", subsec_heading))
    table3_data = [
        [Paragraph("Tested Condition / Module", table_header),
         Paragraph("Parameter Range / Ablation", table_header),
         Paragraph("Baseline / Ablated Outcome", table_header),
         Paragraph("Full TactAI Filter Outcome (Ours)", table_header)],
        [Paragraph("Ablation 1: Invariants", table_cell),
         Paragraph("w/o CoP &Delta;<b>r</b><sub><i>c</i></sub> &amp; Tensor <b>C</b>", table_cell),
         Paragraph("0.0% Insertion (Stalls on Rim at 11.79 N)", table_cell),
         Paragraph("<b>100.0% Succ. (Sub-Degree Alignment)</b>", table_cell)],
        [Paragraph("Ablation 2: Anti-Jamming", table_cell),
         Paragraph("w/o Stage 2 Un-Wedging <i>h</i><sub>jam</sub>", table_cell),
         Paragraph("0.0% Insertion (30.22 &plusmn; 2.02 N Camming)", table_cell),
         Paragraph("<b>100.0% Succ. (0.0% Jamming, &lt;15 N)</b>", table_cell)],
        [Paragraph("Ablation 3: Energy Tank", table_cell),
         Paragraph("w/o Energy Tank Passivity <i>T</i>(<i>t</i>)", table_cell),
         Paragraph("Bench A: 35% Succ. | Bench B: 77.0 N Chatter", table_cell),
         Paragraph("<b>100% Succ. | 10.64 N Chatter-Free</b>", table_cell)],
        [Paragraph("Ablation 4: Slip Barrier", table_cell),
         Paragraph("w/o Incipient Slip CBF <i>h</i><sub>slip</sub>", table_cell),
         Paragraph("100.0% Tool Drop (148.8 mm Gross Slip)", table_cell),
         Paragraph("<b>90.0% Retention (0.44 mm Mean Slip)</b>", table_cell)],
        [Paragraph("Surface Stiffness <i>K</i><sub>env</sub>", table_cell),
         Paragraph("10<sup>2</sup> to 10<sup>5</sup> N/m (<i>N</i> = 5 Seeds)", table_cell),
         Paragraph("0% Succ. | Jammed (41.15&ndash;43.98 N Spikes)", table_cell),
         Paragraph("<b>10<sup>2</sup>: Safe (3.2 N, No Insert); 10<sup>3&ndash;5</sup>: Succ.</b>", table_cell)],
        [Paragraph("Friction &mu; &amp; Latency &tau;", table_cell),
         Paragraph("&mu; &isin; [0.15, 0.85], &tau;<sub>delay</sub> &isin; [10, 50] ms", table_cell),
         Paragraph("0% Succ. | Wedged (43.62&ndash;44.18 N)", table_cell),
         Paragraph("<b>100% Succ. | Safe (8.63&ndash;11.91 N Peaks)</b>", table_cell)]
    ]
    t3 = Table(table3_data, colWidths=[1.65 * inch, 1.55 * inch, 1.95 * inch, 2.00 * inch])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('BACKGROUND', (3, 1), (3, -1), colors.HexColor('#ECFDF5')),
        ('TOPPADDING', (0, 0), (-1, -1), 2.8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.8),
    ]))
    story.append(t3)
    story.append(Spacer(1, 3.5))

    # Figures
    fig1_bench_a = os.path.abspath("docs/figures/phase3_benchmark_comparison.png")
    if os.path.exists(fig1_bench_a):
        story.append(KeepTogether([
            Image(fig1_bench_a, width=6.25 * inch, height=4.30 * inch),
            Paragraph(
                "<b>Fig. 1.</b> Benchmark A Structured Stress Set &amp; Multi-Seed Robustness Results: (A) Insertion success vs. jamming rate across "
                "<i>N</i> = 50 structured trials; (B) Peak contact force vs. initial 2-axis OOD tilt norm ||<b>&theta;</b><sub>0</sub>||<sub>2</sub> (&mu; &plusmn; 1&sigma;); "
                "(C) Surface stiffness sweep across <i>K</i><sub>env</sub> &isin; [10<sup>2</sup>, 10<sup>5</sup>] N/m (<i>N</i> = 5 seeds/point; note safe force but no insertion at 10<sup>2</sup> N/m); and "
                "(D) Tactile perception latency robustness (&tau;<sub>delay</sub> &isin; [10, 50] ms, <i>N</i> = 5 seeds/point). Held-out random results are reported in Table I.",
                caption_style
            )
        ]))

    fig2_bc_ab = os.path.abspath("docs/figures/phase3_benchmarks_bc_ablation.png")
    if os.path.exists(fig2_bc_ab):
        story.append(KeepTogether([
            Image(fig2_bc_ab, width=6.65 * inch, height=2.06 * inch),
            Paragraph(
                "<b>Fig. 2.</b> Benchmarks B &amp; C (<i>N</i> = 20 Each) and Component Ablation Study: (A) Benchmark B non-planar stiff contour force tracking "
                "(<i>K</i><sub>env</sub> = 10<sup>4</sup> N/m, 25 ms delay) showing full oscillation peaks (127.6 N for Wrist F/T Admittance, "
                "77.0 N w/o Energy Tank vs. 10.6 N chatter-free for Ours); (B) Benchmark C mean cumulative tool slip across &mu; &isin; [0.22, 0.58] "
                "(0.44 mm mean slip for Ours vs. 44.0&ndash;148.8 mm for baselines); and (C) Component ablation study on <i>N</i> = 20 OOD trials.",
                caption_style
            )
        ]))

    fig3_sim = os.path.abspath("docs/figures/simulation_3d_overview.png")
    fig4_p2 = os.path.abspath("docs/figures/phase2_validation_benchmark.png")
    if os.path.exists(fig3_sim) and os.path.exists(fig4_p2):
        story.append(KeepTogether([
            Image(fig3_sim, width=5.55 * inch, height=4.14 * inch),
            Paragraph(
                "<b>Fig. 3.</b> MuJoCo 3.15 Simulation Workbench: (1) 7-DoF Franka Emika Panda arm with dual GelSight Mini pads "
                "(<i>K</i><sub>env</sub> = 10<sup>4</sup> N/m); (2) Aligned insertion seating to 20.2 mm; (3) Unconstrained baseline jamming at "
                "+10&deg; OOD tilt (44.1 N spike); and (4) Synthesized 3D GelSight normal pressure field <i>f<sub>z</sub></i>(<i>u</i>,<i>v</i>).",
                caption_style
            )
        ]))
        story.append(KeepTogether([
            Image(fig4_p2, width=5.25 * inch, height=3.88 * inch),
            Paragraph(
                "<b>Fig. 4.</b> Closed-loop time-series telemetry (+10&deg; OOD insertion): (Top) Contact force arrest at 8.78 N (&lt; 15 N CBF limit) vs. "
                "44.05 N unconstrained jamming spike; (Middle) Contact-triggered rotational alignment from +10.0&deg; to 0.001&deg;; and "
                "(Bottom) 2 mm un-wedging micro-retraction followed by 20.16 mm deep slot insertion.",
                caption_style
            )
        ]))

    # ---------------------------------------------------------
    # SECTION VI: DISCUSSION & LIMITATIONS
    # ---------------------------------------------------------
    story.append(Paragraph("VI. DISCUSSION AND LIMITATIONS", sec_heading))
    story.append(Paragraph(
        "<b>Why Tactile Invariants Help:</b> Wrist F/T admittance fails mainly at larger tilts, consistent with the tip-shear moment opposing the true "
        "alignment couple at the wrist. Using &Delta;<b>r</b><sub><i>c</i></sub> (with <b>C</b>) and un-wedging before rotation removes the jams entirely in "
        "our setting. Removing Stage 2 reintroduces camming spikes (30.2 N), and removing the tactile invariants causes rim stall, indicating that both "
        "are necessary here.<br/>"
        "<b>Limitations:</b> (i) <i>Simulation only:</i> no hardware experiments. (ii) The <i>tactile signal is idealized</i> (Sec. V); real FEATS estimates "
        "include noise, hysteresis, and bias, and the latching threshold and &kappa;<sub>min</sub> floor are untested against them. (iii) The policy is a "
        "<i>proxy</i>, not trained FARM. (iv) Several gains and the Stage 3 rim-arrest guard are <i>hand-tuned</i>; we mitigate this with a held-out evaluation "
        "but have not run a sensitivity analysis. (v) Proposition 1 is a sufficient-condition sketch, not a formal proof. (vi) At very soft surfaces "
        "(10<sup>2</sup> N/m) the filter remains safe but does not insert. (vii) Out-of-basin trials require active search (e.g., spiral search), which we do not implement.<br/>"
        "<b>Conclusion:</b> A supervisory tactile filter can eliminate jamming and enforce a contact-force limit around an unmodified action-chunking policy "
        "in simulation, while substantially improving insertion success over wrist F/T admittance. Hardware validation with learned tactile estimation is the "
        "necessary next step.",
        body_style
    ))

    # ---------------------------------------------------------
    # REFERENCES
    # ---------------------------------------------------------
    story.append(Paragraph("REFERENCES", sec_heading))
    refs = [
        "[1] E. Helmut, N. Funk, T. Schneider, C. de Farias, and J. Peters, 'Tactile-Conditioned Diffusion Policy for Force-Aware Robotic Manipulation,' arXiv:2510.13324, 2025.",
        "[2] E. Helmut, L. Dziarski, N. Funk, B. Belousov, and J. Peters, 'Learning Force Distribution Estimation for the GelSight Mini Optical Tactile Sensor Based on Finite Element Analysis,' arXiv preprint, 2024.",
        "[3] D. H. Nguyen, T. Schneider, G. Duret, A. Kshirsagar, B. Belousov, and J. Peters, 'TacEx: GelSight Tactile Simulation in Isaac Sim,' arXiv:2411.04776, 2024.",
        "[4] A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, and P. Tabuada, 'Control Barrier Functions: Theory and Applications,' in Proc. European Control Conf. (ECC), pp. 3420–3431, 2019.",
        "[5] F. Ferraguti, C. Secchi, and C. Fantuzzi, 'A Tank-Based Approach to Impedance Control with Variable Stiffness,' in Proc. IEEE Int. Conf. Robotics and Automation (ICRA), pp. 4948–4953, 2013.",
        "[6] D. E. Whitney, 'Quasi-Static Assembly of Compliantly Supported Rigid Parts,' ASME Journal of Dynamic Systems, Measurement, and Control, vol. 104, no. 1, pp. 65–77, 1982.",
        "[7] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, 'Diffusion Policy: Visuomotor Policy Learning via Action Diffusion,' in Proc. Robotics: Science and Systems (RSS), 2023.",
        "[8] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, 'Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware,' in Proc. Robotics: Science and Systems (RSS), 2023."
    ]
    ref_style = ParagraphStyle(
        'RefItem',
        parent=body_style,
        fontName='Times-Roman',
        fontSize=7.7,
        leading=9.7,
        spaceAfter=2.0
    )
    for r in refs:
        story.append(Paragraph(r, ref_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated peer-reviewed IEEE (ieeeconf) preprint PDF at: {output_path}")


if __name__ == "__main__":
    create_paper_pdf()
