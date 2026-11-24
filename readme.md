<div align="center"><img src="https://img.shields.io/badge/Code-Coming%20Soon-blue?style=for-the-badge&logo=github" alt="Code coming soon">
<img src="https://img.shields.io/badge/Benchmark-LIBERO-7953AF?style=for-the-badge" alt="LIBERO">
<img src="https://img.shields.io/badge/Poisoning%20Rate-1%25-C9515B?style=for-the-badge" alt="1% poisoning">🎯 Sparse Poison, Strong Hijack
Efficient Trajectory-Level Backdoors in Vision-Language-Action Models with Only 1% Poisoning
<p align="center">
  <strong>Simin Huang*</strong><sup>1</sup> &nbsp;·&nbsp;
  <strong>Jiawei Chen*</strong><sup>1,2</sup> &nbsp;·&nbsp;
  <strong>Shaolong Zhu</strong><sup>3</sup> &nbsp;·&nbsp;
  <strong>Qiwei Liang</strong><sup>4</sup> &nbsp;·&nbsp;
  <strong>Zhaoxia Yin†</strong><sup>1</sup>
</p><p align="center">
  <sup>1</sup>East China Normal University &nbsp;·&nbsp;
  <sup>2</sup>Zhongguancun Academy<br>
  <sup>3</sup>Tsinghua University &nbsp;·&nbsp;
  <sup>4</sup>The Hong Kong University of Science and Technology (Guangzhou)
</p><p align="center">
  <em>* Equal contribution &nbsp; † Corresponding author</em>
</p><p align="center">
  📄 Paper (Coming Soon) &nbsp;|&nbsp;
  <a href="#-main-results">📊 Results</a> &nbsp;|&nbsp;
  <a href="#-citation">📖 Citation</a>
</p><!-- Replace the paper placeholder with the public paper URL when available. -->
<!-- Optional teaser: <img src="assets/teaser.png" width="95%" alt="Trajectory-level backdoors under sparse poisoning"> --></div>📢 What's New!
- [2026] The repository for Sparse Poison, Strong Hijack is initialized. Code and experimental instructions are coming soon.
📌 Abstract
Vision-Language-Action (VLA) models advance robotic manipulation, but their reliance on demonstrations exposes them to backdoor threats. Unlike attacks that merely disrupt task execution, trajectory-level targeted backdoors require sustained closed-loop redirection to complete an attacker-specified multi-step task. Existing methods often rely on high poisoning rates, compromising clean-task performance, while sparse poisoning provides insufficient target-trajectory supervision and leaves backdoor learning vulnerable to interference from dominant clean-task updates.

We propose a two-stage co-optimization framework guided by vision, language, and action:
- Multimodal-Guided Differentiable 3D Trigger Optimization — addresses scarce target-trajectory supervision through multimodal guidance.
- Conflict-Aware Dual-LoRA Adaptation and Fusion — addresses interference between clean and backdoor learning and consolidates the adaptations into a unified policy.
Experiments across four VLA architectures and four LIBERO suites demonstrate consistent improvements over existing targeted backdoor baselines. With only 1% poisoned demonstrations, our method achieves 86.4% Target Success Rate (TSR) and 95.2% Clean Success Rate (CSR) on OpenVLA-OFT in LIBERO-Spatial. Physical robot experiments further demonstrate trajectory-level redirection.
🏗️ Framework Overview
<!-- Add Figure 2 from the paper after uploading the image asset:
<p align="center">
  <img src="assets/framework.png" width="95%" alt="Overview of the two-stage framework">
</p>
-->The framework addresses two complementary challenges in trajectory-level backdoor learning: limited target-trajectory supervision and interference between clean and backdoor objectives.
Stage	Focus
Stage I: Trigger Efficacy Optimization	Multimodal supervision and differentiable 3D rendering
Stage II: Conflict-Aware Dual-LoRA Adaptation	Objective-specific adaptation and calibrated parameter fusion


📈 Main Results
Trajectory-Level Targeted Backdoors at 1% Poisoning
Results are averaged across LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-Long. Each entry reports CSR / TSR (%); higher is better for both metrics.
Model	Clean-only	GoBA	BackdoorVLA-VT	Ours
OpenVLA	76.5 / 0.0	69.1 / 7.7	68.4 / 10.6	72.2 / 75.3
OpenVLA-OFT	97.1 / 0.0	81.5 / 17.2	81.3 / 16.8	93.2 / 79.0
π₀	94.2 / 0.0	89.8 / 13.3	85.9 / 17.9	91.1 / 78.3
π₀.₅	96.9 / 0.0	89.4 / 13.9	90.6 / 16.6	93.2 / 80.3


Bold denotes the highest CSR and TSR among the evaluated backdoor methods. Clean-only CSR is taken from published evaluations, whose protocols may differ.

Across all 16 model–suite settings, our method achieves 78.2% average TSR, surpassing the strongest baseline by 62.7 percentage points, with the smallest clean-task degradation.
Real-World Evaluation
We evaluate a π₀.₅-adapted policy on an AgileX PiPER 6-DoF robotic arm equipped with an Intel RealSense D435i camera across four tabletop manipulation tasks.
Task	Clean-only CSR (%)	Ours CSR (%)	Ours TSR (%)	Ours UASR (%)
T1 — Spatial	75.0	70.0	60.0	80.0
T2 — Object	70.0	70.0	40.0	100.0
T3 — Goal	70.0	70.0	60.0	90.0
T4 — Long	35.0	30.0	70.0	90.0
Total	62.5	60.0	57.5	90.0


The physical evaluation comprises 80 clean and 80 triggered trials, demonstrating the feasibility of trajectory-level redirection under real-world perception and control noise.
<!-- Optional: add simulation and real-robot trajectory visualization figures here. -->🧪 Evaluation Protocol
The simulation experiments cover four VLA models (OpenVLA, OpenVLA-OFT, π₀, and π₀.₅) and four LIBERO suites (Spatial, Object, Goal, and Long). Each model–suite setting uses 500 clean and 500 triggered closed-loop episodes, with 50 trials per task under each condition.
Metric	Definition
CSR ↑	Original-task completion rate without the trigger
TSR ↑	Target-task completion rate with the trigger
UASR ↑	Original-task failure rate with the trigger


TSR measures complete target-task execution rather than isolated action deviations or failures of the original task.
🚀 Code Availability
The implementation and experimental instructions are coming soon.
<!-- Add repository structure, environment setup, data preparation, and execution commands after the code release. -->📖 Citation
If you find our work useful for your research, please cite:
@misc{huang2026sparsepoison,
  title  = {Sparse Poison, Strong Hijack: Efficient Trajectory-Level Backdoors in Vision-Language-Action Models with Only 1\% Poisoning},
  author = {Simin Huang and Jiawei Chen and Shaolong Zhu and Qiwei Liang and Zhaoxia Yin},
  year   = {2026}
}
<!-- Update the BibTeX entry when public publication metadata becomes available. -->🙏 Acknowledgements
We thank the authors of LIBERO, OpenVLA, OpenVLA-OFT, OpenPI, nvdiffrast, and MuJoCo for their open-source contributions that made this research possible.
<div align="center">
  <sub>⭐ If you find this work useful, please consider starring the repository!</sub>
</div>
