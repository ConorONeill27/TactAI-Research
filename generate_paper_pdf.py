"""
Generate IEEE/RSS Research Paper PDF for Tac-Guard
Title: Tac-Guard: Tactile Contact-Manifold Safety Filtering for Learned Manipulation Policies
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

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
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "Tac-Guard: Tactile Safety Filtering for Learned Manipulation Policies")
            self.drawRightString(8.5 * inch - 54, 11 * inch - 36, "Tact AI -- Research Draft (October 2026)")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 40, 8.5 * inch - 54, 11 * inch - 40)
            
        # Footer
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(8.5 * inch - 54, 36, page_str)
        self.drawString(54, 36, "CONFIDENTIAL -- PREPARED FOR RESEARCH COLLABORATION AT IAS TU DARMSTADT")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 46, 8.5 * inch - 54, 46)
        self.restoreState()


def create_paper_pdf(output_path="D:/TactAI/tac_guard_research_paper.pdf"):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'PaperTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        alignment=1,  # Centered
        textColor=colors.HexColor('#111827'),
        spaceAfter=8
    )
    
    subtitle_style = ParagraphStyle(
        'PaperSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=14,
        alignment=1,
        textColor=colors.HexColor('#374151'),
        spaceAfter=12
    )
    
    authors_style = ParagraphStyle(
        'PaperAuthors',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        alignment=1,
        textColor=colors.HexColor('#1F2937'),
        spaceAfter=4
    )
    
    affil_style = ParagraphStyle(
        'PaperAffil',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=12,
        alignment=1,
        textColor=colors.HexColor('#4B5563'),
        spaceAfter=14
    )
    
    abstract_heading = ParagraphStyle(
        'AbstractHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=12,
        textColor=colors.HexColor('#111827'),
        spaceAfter=4
    )
    
    abstract_body = ParagraphStyle(
        'AbstractBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        alignment=4,  # Justified
        textColor=colors.HexColor('#1F2937')
    )
    
    sec_heading = ParagraphStyle(
        'SecHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    subsec_heading = ParagraphStyle(
        'SubSecHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        alignment=4,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=6
    )
    
    math_style = ParagraphStyle(
        'MathBlock',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=8.5,
        leading=11.5,
        alignment=1,
        textColor=colors.HexColor('#0F172A'),
        backColor=colors.HexColor('#F8FAFC'),
        borderColor=colors.HexColor('#E2E8F0'),
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=6,
        spaceAfter=6
    )
    
    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#0369A1')
    )
    
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        alignment=1,
        textColor=colors.HexColor('#1F2937')
    )
    
    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        alignment=1,
        textColor=colors.white
    )

    story = []
    
    # ---------------------------------------------------------
    # TITLE & HEADER
    # ---------------------------------------------------------
    story.append(Paragraph("Tac-Guard: Tactile Contact-Manifold Safety Filtering for Learned Manipulation Policies", title_style))
    story.append(Paragraph("A 1 kHz Control Barrier Function Architecture for Anti-Jamming and Guaranteed Contact Passivity", subtitle_style))
    story.append(Paragraph("Tact AI Research Group<sup>1</sup>, In Collaboration with Institute for Intelligent Autonomous Systems (IAS)<sup>2</sup>", authors_style))
    story.append(Paragraph("<sup>1</sup>Autonomous Physical AI Initiative &nbsp;&nbsp;|&nbsp;&nbsp; <sup>2</sup>Technical University of Darmstadt, Germany", affil_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=10))
    
    # ---------------------------------------------------------
    # ABSTRACT BOX
    # ---------------------------------------------------------
    abstract_text = (
        "<b><i>Abstract</i>—Learned robotic manipulation policies (such as diffusion policies and imitation learning models) "
        "have achieved impressive visual versatility but frequently suffer from catastrophic force spikes, mechanical wedging, "
        "and part damage when deployed in contact-rich tasks under geometric misalignments. Because neural policies are unconstrained "
        "approximators, out-of-distribution (OOD) contact states lead to dangerous extrapolation and task failure. In this work, "
        "we present Tac-Guard, a high-frequency (1 kHz) supervisory control architecture that sits between any pre-trained "
        "manipulation policy and robot actuators to guarantee physical contact safety. Tac-Guard processes high-resolution 3D force "
        "distribution fields from vision-based elastomeric tactile sensors (GelSight Mini) and extracts invariant geometric contact "
        "manifolds: the Traction Center of Pressure (CoP), the Contact Constraint Tensor, and the Incipient Slip Ratio. "
        "Using these invariants, Tac-Guard solves a lightweight convex Quadratic Program (QP) subject to Control Barrier Functions (CBFs) "
        "and an Energy Tank passivity monitor. This formulation mathematically bounds contact forces below material failure thresholds, "
        "prevents two-point contact force couples (jamming), and enforces asymptotic Lyapunov passivity during non-smooth contact "
        "transitions. Crucially, Tac-Guard requires zero retraining or additional demonstrations from the underlying policy. "
        "We specify the mathematical architecture, formal safety theorems, and a standardized benchmark protocol across tight-clearance "
        "insertion, non-planar contour wiping, and fragile object manipulation.</b>"
    )
    
    abs_table = Table(
        [[Paragraph(abstract_text, abstract_body)]],
        colWidths=[7.0 * inch]
    )
    abs_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F1F5F9')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#94A3B8')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(abs_table)
    story.append(Spacer(1, 10))
    
    # ---------------------------------------------------------
    # SECTION I: INTRODUCTION
    # ---------------------------------------------------------
    story.append(Paragraph("I. INTRODUCTION", sec_heading))
    story.append(Paragraph(
        "Modern deep robot learning has made remarkable strides in learning complex manipulation behaviors directly from "
        "multimodal observations. Architectures such as Diffusion Policies (Chi et al., 2023) and Action Chunking Transformers "
        "(ACT; Zhao et al., 2023) excel at imitating human teleoperation across broad visual scenes. Recently, frameworks such as "
        "FARM (Helmut et al., ICRA 2026) have integrated continuous force distribution estimates from optical tactile sensors "
        "(FEATS; Helmut et al., 2024) into diffusion policies to enable force-conditioned robotic interaction.",
        body_style
    ))
    story.append(Paragraph(
        "However, a fundamental vulnerability plagues all unconstrained neural policies in contact-rich settings: "
        "<b>the absence of hard physical safety guarantees</b>. When a grasped tool or workpiece encounters an unexpected "
        "geometric constraint (e.g., a peg misaligned by 5 degrees with respect to a 0.2 mm clearance slot), the observation becomes "
        "out-of-distribution (OOD). Continuous neural networks cannot extrapolate non-smooth contact boundaries; they attempt to "
        "interpolate between demonstration modes, frequently commanding full-power actuator commands directly into rigid obstacles. "
        "In stiff environments (stiffness K > 10,000 N/m), this causes explosive force spikes within 20 milliseconds, destroying "
        "tactile elastomer membranes, stalling joint drives, or wedging mechanical assemblies permanently.",
        body_style
    ))
    story.append(Paragraph(
        "Historically, roboticists attempted to resolve this through pure passive mechanics (Remote Center of Compliance devices; "
        "Whitney, 1982) or low-gain isotropic Cartesian impedance control. However, fixed low-stiffness control severely degrades "
        "free-space tracking accuracy and cannot distinguish between desired task force and jamming resistance. Conversely, "
        "retraining diffusion models with negative failure samples requires hundreds of destructive trials on real hardware.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Core Contribution:</b> In this work, we propose <b>Tac-Guard</b>, an optimization-based supervisory safety filter. "
        "Rather than retraining the primary policy, Tac-Guard intercepts policy action commands at 1 kHz and solves a convex "
        "Quadratic Program (QP) with Control Barrier Functions (CBFs). By exploiting the dense spatial force distribution field "
        "of the GelSight Mini sensor, Tac-Guard computes contact normals, incipient slip ratios, and jamming force couples in real time, "
        "projecting the nominal command onto the nearest physically safe action. The primary contributions are:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Tactile Invariant Manifold Extraction:</b> A real-time formulation mapping 2D traction distributions to the Contact Constraint Tensor and Incipient Slip Margin.", body_style))
    story.append(Paragraph("&bull; <b>Anti-Jamming & Damage CBFs:</b> Analytical control barrier functions that provably prevent two-point wedging and normal force violations.", body_style))
    story.append(Paragraph("&bull; <b>Energy Tank Passivity Guarantee:</b> An adaptive tank architecture that eliminates contact chattering on stiff surfaces under neural perception latency.", body_style))
    story.append(Paragraph("&bull; <b>Zero-Retraining Plug-and-Play Integration:</b> Direct deployment wrapping around pre-trained diffusion policies or teleoperation streams.", body_style))
    
    # ---------------------------------------------------------
    # SECTION II: RELATED WORK
    # ---------------------------------------------------------
    story.append(Paragraph("II. RELATED WORK", sec_heading))
    story.append(Paragraph(
        "<b>Tactile Sensing and Force Distribution:</b> High-resolution optical tactile sensors (GelSight, DIGIT) provide rich geometric "
        "contact imprints. Recent breakthroughs at IAS TU Darmstadt, notably FEATS (Helmut, Belousov, Peters, 2024), demonstrated "
        "that raw GelSight Mini images can be mapped in real time (25 Hz) to 3D continuous force distribution fields using a U-Net trained "
        "on CalculiX Finite Element Analysis. TacEx (Nguyen et al., 2024) established soft-body elastomeric simulation in Isaac Sim. "
        "However, existing literature treats these fields primarily as perceptual observations rather than control-theoretic invariants.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Imitation Learning & Tactile Diffusion Policies:</b> FARM (Helmut, Funk, Peters, ICRA 2026) demonstrated that diffusion "
        "policies conditioned on FEATS representations achieve superior compliance compared to vision-only baselines. However, like all "
        "diffusion models, FARM lacks constraint guarantees and remains vulnerable to OOD jamming. Recent safety-guided diffusion works "
        "(e.g., PACT; 2025) focus on kinematic state-space bounds but do not handle dynamic contact wrench constraints or soft elastomer slip.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Control Barrier Functions & Passivity in Robotics:</b> Control Barrier Functions (Ames et al., 2019) have emerged as the "
        "mathematical gold standard for safety-critical control. Energy Tanks (Ferraguti et al., 2013; Kronander & Billard, 2014) ensure "
        "strict passivity in variable impedance systems. Tac-Guard synthesizes these principles with distributed tactile contact mechanics.",
        body_style
    ))
    
    # ---------------------------------------------------------
    # SECTION III: PROBLEM FORMULATION & CONTACT MECHANICS
    # ---------------------------------------------------------
    story.append(Paragraph("III. PROBLEM FORMULATION & CONTACT MECHANICS", sec_heading))
    story.append(Paragraph(
        "Consider a fully actuated robotic manipulator with state x = [q, q_dot] in R^{2n} and control input u = tau_cmd in R^n. "
        "The end-effector is equipped with an elastomeric tactile sensor contacting an environment. The contact interaction generates "
        "a spatial traction field f(u, v) = [f_x(u, v), f_y(u, v), f_z(u, v)]^T over the sensor gel patch Omega.",
        body_style
    ))
    story.append(Paragraph("<b>Definition 1 (Tactile Constraint Tensor):</b> Let r_c be the Center of Pressure. The spatial contact constraint tensor C in R^{2x2} is defined as:", subsec_heading))
    story.append(Paragraph("C = integral_Omega ([u, v]^T - r_c)([u, v]^T - r_c)^T * f_z(u, v) du dv", math_style))
    story.append(Paragraph(
        "The eigenvalues of C indicate the spatial dispersion of contact pressure, while its eigenvectors define the orientation of "
        "environmental contact edges and slot boundaries.",
        body_style
    ))
    story.append(Paragraph("<b>Definition 2 (Incipient Slip Ratio):</b> The local slip margin rho(u, v) is given by Coulomb friction. The global Stick Ratio rho_stick in [0, 1] is:", subsec_heading))
    story.append(Paragraph("rho_stick = (integral_Omega I(||f_xy|| < mu * f_z) du dv) / (integral_Omega I(f_z > 0) du dv)", math_style))
    story.append(Paragraph(
        "When rho_stick -> 1, the patch is fully adhering; when rho_stick -> 0, macroscopic slip occurs.",
        body_style
    ))
    story.append(Paragraph("<b>Definition 3 (Jamming Invariant):</b> For an insertion depth L and contact force couple, two-point jamming occurs when the ratio of insertion force to tilting torque falls below the critical threshold:", subsec_heading))
    story.append(Paragraph("Ratio_jam = F_insert / (||tau_tilt|| + eps) <= 1 / mu_env  ==>  JAMMED", math_style))
    
    # ---------------------------------------------------------
    # SECTION IV: TAC-GUARD ARCHITECTURE & CBF FORMULATION
    # ---------------------------------------------------------
    story.append(Paragraph("IV. TAC-GUARD SAFETY FILTER ARCHITECTURE", sec_heading))
    story.append(Paragraph(
        "Let u_pi in R^n be the nominal torque command generated by an unconstrained policy (e.g., FARM diffusion policy). "
        "Tac-Guard computes a safe filtered command u_safe = u* by solving a convex Quadratic Program (QP) at 1 kHz:",
        body_style
    ))
    
    story.append(Paragraph(
        "min_{u} (1/2) * ||u - u_pi||_W^2<br/>"
        "s.t.  L_f h_force(x) + L_g h_force(x) u + alpha_1(h_force(x)) >= 0  (Force Limit CBF)<br/>"
        "      L_f h_jam(x) + L_g h_jam(x) u + alpha_2(h_jam(x)) >= 0      (Anti-Jamming CBF)<br/>"
        "      L_f h_slip(x) + L_g h_slip(x) u + alpha_3(h_slip(x)) >= 0    (Slip Prevention CBF)<br/>"
        "      (1/2) * q_dot^T * dK/dt * q_dot <= P_tank_available           (Passivity Energy Tank)",
        math_style
    ))
    
    story.append(Paragraph(
        "<b>Theorem 1 (Forward Invariance & Zero Jamming):</b> If the safe set C_safe = {x | h_i(x) >= 0} is non-empty and "
        "alpha_i are class-K functions, the closed-loop system under Tac-Guard guarantees that x(t) in C_safe for all t >= 0. "
        "Consequently, peak normal forces satisfy F_n(t) <= F_max and the manipulator cannot enter the two-point jamming regime.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Energy Tank Stability:</b> Because tactile perception runs at 25-50 Hz while torque control runs at 1 kHz, latency can "
        "generate virtual energy on stiff contact surfaces. By augmenting the state with a scalar energy tank T(t), any stiffness "
        "variation is constrained by the tank balance, guaranteeing strict passivity regardless of perception latency.",
        body_style
    ))
    
    # ---------------------------------------------------------
    # SECTION V: EXPERIMENTAL BENCHMARK PROTOCOL
    # ---------------------------------------------------------
    story.append(Paragraph("V. EXPERIMENTAL BENCHMARK PROTOCOL", sec_heading))
    story.append(Paragraph(
        "To rigorously validate Tac-Guard prior to physical deployment at IAS TU Darmstadt, we establish three standardized "
        "contact-rich manipulation benchmarks simulated in MuJoCo 3.15 and TacEx / Isaac Sim with GelSight Mini sensor models:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Benchmark A (Sub-Millimeter Slot Insertion):</b> Workpiece insertion with 0.15 mm to 0.30 mm clearance under out-of-distribution initial tilts (+- 15 degrees).", body_style))
    story.append(Paragraph("&bull; <b>Benchmark B (Non-Planar Stiff Contour Tracking):</b> End-effector gliding across an unknown curved surface of stiffness K_env = 10^4 N/m at 0.05 m/s.", body_style))
    story.append(Paragraph("&bull; <b>Benchmark C (In-Hand Controlled Extrinsic Manipulation):</b> Reorienting a tool against gravity while preventing drop using the incipient slip CBF.", body_style))
    
    # Placeholder Tables for Research Results
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>TABLE I: Benchmark A Performance (0.2 mm Clearance Insertion under +-15 deg Tilt)</b>", subsec_heading))
    
    table1_data = [
        [Paragraph("Controller / Policy", table_header),
         Paragraph("Success Rate (%)", table_header),
         Paragraph("Peak Force (N)", table_header),
         Paragraph("Jamming Rate (%)", table_header),
         Paragraph("Cycle Time (s)", table_header)],
        [Paragraph("Unconstrained Diffusion Policy (FARM)", table_cell),
         Paragraph("[TO BE EVAL]", table_cell),
         Paragraph("[TO BE EVAL]", table_cell),
         Paragraph("[TO BE EVAL]", table_cell),
         Paragraph("[TO BE EVAL]", table_cell)],
        [Paragraph("Wrist F/T Admittance Filter", table_cell),
         Paragraph("[TO BE EVAL]", table_cell),
         Paragraph("[TO BE EVAL]", table_cell),
         Paragraph("[TO BE EVAL]", table_cell),
         Paragraph("[TO BE EVAL]", table_cell)],
        [Paragraph("Fixed Isotropic Impedance (K=200)", table_cell),
         Paragraph("[TO BE EVAL]", table_cell),
         Paragraph("[TO BE EVAL]", table_cell),
         Paragraph("[TO BE EVAL]", table_cell),
         Paragraph("[TO BE EVAL]", table_cell)],
        [Paragraph("<b>Tac-Guard + FARM (Ours)</b>", table_cell),
         Paragraph("<b>[TARGET: 100%]</b>", table_cell),
         Paragraph("<b>[TARGET: <15 N]</b>", table_cell),
         Paragraph("<b>[TARGET: 0.0%]</b>", table_cell),
         Paragraph("<b>[TARGET: <2.5s]</b>", table_cell)]
    ]
    t1 = Table(table1_data, colWidths=[2.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#F8FAFC')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t1)
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("<b>TABLE II: Robustness Across Environmental Stiffness & Friction Variations</b>", subsec_heading))
    table2_data = [
        [Paragraph("Environmental Parameter", table_header),
         Paragraph("Tested Range", table_header),
         Paragraph("Baseline FARM Status", table_header),
         Paragraph("Tac-Guard Filter Status", table_header)],
        [Paragraph("Surface Stiffness K_env", table_cell),
         Paragraph("10^2 to 10^5 N/m", table_cell),
         Paragraph("[TO BE EVAL: Chattering/Spikes]", table_cell),
         Paragraph("<b>[TARGET: Passive / Stable]</b>", table_cell)],
        [Paragraph("Friction Coefficient mu", table_cell),
         Paragraph("0.15 to 0.85", table_cell),
         Paragraph("[TO BE EVAL: Slippage / Stalling]", table_cell),
         Paragraph("<b>[TARGET: Invariant Slip Margin]</b>", table_cell)],
        [Paragraph("Perception Latency", table_cell),
         Paragraph("10 ms to 50 ms", table_cell),
         Paragraph("[TO BE EVAL: Instability / Limit Cycles]", table_cell),
         Paragraph("<b>[TARGET: Passivity Preserved]</b>", table_cell)]
    ]
    t2 = Table(table2_data, colWidths=[2.2 * inch, 1.4 * inch, 1.7 * inch, 1.7 * inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#F8FAFC')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t2)
    
    # ---------------------------------------------------------
    # SECTION VI: DISCUSSION & SIM-TO-REAL ROADMAP
    # ---------------------------------------------------------
    story.append(Spacer(1, 8))
    story.append(Paragraph("VI. DISCUSSION & SIM-TO-REAL ROADMAP", sec_heading))
    story.append(Paragraph(
        "<b>Zero Sim-to-Real Domain Gap:</b> Unlike visual perception networks which suffer from photometric discrepancies between "
        "renderers and physical cameras, Tac-Guard operates directly on physical SI units: force distribution fields (N/cm^2), "
        "contact normals, and stick ratios. Consequently, the safety QP formulated in simulation transfers to real hardware with zero "
        "retraining or domain randomization. Upon arrival at IAS TU Darmstadt in January 2027, Tac-Guard will be compiled into "
        "real-time C++ (using qpOASES / OSQP) and integrated into the Franka Control Interface (FCI / libfranka) at 1 kHz.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Conclusion:</b> Tac-Guard bridges the gap between deep robot learning and rigorous control theory. By providing "
        "provable physical safety filters grounded in distributed tactile force fields, it unlocks reliable, damage-free "
        "deployment of foundation policies in real-world contact-rich automation.",
        body_style
    ))
    
    # REFERENCES
    story.append(Paragraph("REFERENCES", sec_heading))
    refs = [
        "[1] E. Helmut, N. Funk, T. Schneider, C. de Farias, and J. Peters, 'Tactile-Conditioned Diffusion Policy for Force-Aware Robotic Manipulation (FARM),' arXiv:2510.13324, accepted at IEEE ICRA, 2026.",
        "[2] E. Helmut, L. Dziarski, N. Funk, B. Belousov, and J. Peters, 'Learning Force Distribution Estimation for the GelSight Mini Optical Tactile Sensor Based on Finite Element Analysis (FEATS),' NeurIPS Touch Processing Workshop 2024 / IEEE IROS, 2025.",
        "[3] D. H. Nguyen, T. Schneider, G. Duret, A. Kshirsagar, B. Belousov, and J. Peters, 'TacEx: A Modular Tactile Simulation Framework in Isaac Sim,' arXiv:2411.04776, 2024.",
        "[4] A. D. Ames, S. Coogan, M. Egerstedt, M. Notomista, K. Sreenath, and P. Tabuada, 'Control Barrier Functions: Theory and Applications,' European Control Conference (ECC), 2019.",
        "[5] F. Ferraguti, C. Secchi, and C. Fantuzzi, 'A Passivity-Based Strategy for Energy-Efficient Variable Impedance Control,' IEEE Transactions on Robotics, vol. 29, no. 5, 2013.",
        "[6] D. E. Whitney, 'Quasi-static Assembly of Compliantly Supported Rigid Parts,' ASME Journal of Dynamic Systems, Measurement, and Control, vol. 104, no. 1, 1982.",
        "[7] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, 'Diffusion Policy: Visuomotor Policy Learning via Action Diffusion,' Robotics: Science and Systems (RSS), 2023."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle('RefStyle', parent=styles['Normal'], fontName='Helvetica', fontSize=7.5, leading=10, textColor=colors.HexColor('#475569'), spaceAfter=3)))
        
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated research paper PDF at: {output_path}")

if __name__ == "__main__":
    create_paper_pdf()
