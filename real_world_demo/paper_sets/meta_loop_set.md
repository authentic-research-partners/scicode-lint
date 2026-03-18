# Meta Loop Paper Set (Prefilter Run 4)

Used for pattern refinement via feedback. Verification results from this set were used to identify false positive patterns and improve detection questions. DO NOT use for final accuracy testing.

## Metadata

- **Prefilter run:** 4
- **Analysis runs:** 53, 54, 55, 56, 62, 65
- **Papers:** 38
- **Date:** 2026-03-14

## Data Funnel

| Stage | Count | Note |
|-------|-------|------|
| Papers sampled | 38 | created before seeded sampling was added |
| Repos with qualifying files | 47 | Python files with ML imports |
| Qualifying files | 884 | |
| **Papers with self-contained files** | **32** | 6 papers had only fragments |
| Repos with self-contained files | 35 | 12 repos had only fragments |
| Self-contained files | 120 | 14% of qualifying files |
| Files with findings (run 56) | 90 | 75% of analyzed files |
| Clean files (no findings) | 30 | 25% of analyzed files |
| Papers with any findings | 27 | 84% of papers with self-contained files |
| **Papers with verified real bugs** | **24** | **75% of papers with self-contained files** |

16% of papers (6/38) had no self-contained files after prefiltering.

## Results (Run 56)

Overall precision: **45.2%** (99 valid / 219 findings)

### By Domain

| Domain | Papers Analyzed | Findings | Valid | Precision |
|--------|----------------|----------|-------|-----------|
| chemistry | 3 | 60 | 36 | 60.0% |
| earth_science | 4 | 46 | 16 | 34.8% |
| materials | 1 | 22 | 8 | 36.4% |
| none (general ML) | 2 | 19 | 13 | 68.4% |
| economics | 2 | 19 | 4 | 21.1% |
| biology | 4 | 14 | 8 | 57.1% |
| medicine | 2 | 13 | 5 | 38.5% |
| engineering | 3 | 13 | 4 | 30.8% |
| astronomy | 3 | 7 | 2 | 28.6% |
| neuroscience | 2 | 4 | 1 | 25.0% |
| physics | 1 | 2 | 2 | 100.0% |

## Papers

| Domain | Title | URL |
|--------|-------|-----|
| astronomy | CLAP. I. Resolving miscalibration for deep learning-based galaxy photometric redshift estimation | https://paperswithcode.com/paper/clap-i-resolving-miscalibration-for-deep |
| astronomy | Conditional Density Estimation Tools in Python and R with Applications to Photometric Redshifts and Likelihood-Free Cosmological Inference | https://paperswithcode.com/paper/conditional-density-estimation-tools-in |
| astronomy | Determination of galaxy photometric redshifts using Conditional Generative Adversarial Networks (CGANs) | https://paperswithcode.com/paper/determination-of-galaxy-photometric-redshifts |
| astronomy | Solar Flare Forecast: A Comparative Analysis of Machine Learning Algorithms for Solar Flare Class Prediction | https://paperswithcode.com/paper/solar-flare-forecast-a-comparative-analysis |
| biology | C-Norm: a neural approach to few-shot entity normalization | https://paperswithcode.com/paper/c-norm-a-neural-approach-to-few-shot-entity |
| biology | DynamicDTA: Drug-Target Binding Affinity Prediction Using Dynamic Descriptors and Graph Representation | https://paperswithcode.com/paper/dynamicdta-drug-target-binding-affinity |
| biology | ReactEmbed: A Cross-Domain Framework for Protein-Molecule Representation Learning via Biochemical Reaction Networks | https://paperswithcode.com/paper/reactembed-a-cross-domain-framework-for |
| biology | Towards 3D Molecule-Text Interpretation in Language Models | https://paperswithcode.com/paper/towards-3d-molecule-text-interpretation-in |
| chemistry | Coarse Graining Molecular Dynamics with Graph Neural Networks | https://paperswithcode.com/paper/coarse-graining-molecular-dynamics-with-graph |
| chemistry | MassFormer: Tandem Mass Spectrum Prediction for Small Molecules using Graph Transformers | https://paperswithcode.com/paper/massformer-tandem-mass-spectrum-prediction |
| chemistry | Modeling Diverse Chemical Reactions for Single-step Retrosynthesis via Discrete Latent Variables | https://paperswithcode.com/paper/modeling-diverse-chemical-reactions-for |
| chemistry | Preference Optimization for Molecule Synthesis with Conditional Residual Energy-based Models | https://paperswithcode.com/paper/preference-optimization-for-molecule |
| earth_science | GenCast: Diffusion-based ensemble forecasting for medium-range weather | https://paperswithcode.com/paper/gencast-diffusion-based-ensemble-forecasting |
| earth_science | Probabilistic Emulation of a Global Climate Model with Spherical DYffusion | https://paperswithcode.com/paper/probabilistic-emulation-of-a-global-climate |
| earth_science | STConvS2S: Spatiotemporal Convolutional Sequence to Sequence Network for Weather Forecasting | https://paperswithcode.com/paper/stconvs2s-spatiotemporal-convolutional |
| earth_science | WeatherMesh-3: Fast and accurate operational global weather forecasting | https://paperswithcode.com/paper/weathermesh-3-fast-and-accurate-operational |
| economics | Addressing Prediction Delays in Time Series Forecasting: A Continuous GRU Approach with Derivative Regularization | https://paperswithcode.com/paper/addressing-prediction-delays-in-time-series |
| economics | Identity Inference on Blockchain using Graph Neural Network | https://paperswithcode.com/paper/identity-inference-on-blockchain-using-graph |
| economics | Non-Stationary Time Series Forecasting Based on Fourier Analysis and Cross Attention Mechanism | https://paperswithcode.com/paper/non-stationary-time-series-forecasting-based |
| engineering | Diffusion Models Beat GANs on Topology Optimization | https://paperswithcode.com/paper/topodiff-a-performance-and-constraint-guided |
| engineering | HoliCity: A City-Scale Data Platform for Learning Holistic 3D Structures | https://paperswithcode.com/paper/holicity-a-city-scale-data-platform-for |
| engineering | Neural Dual Contouring | https://paperswithcode.com/paper/neural-dual-contouring |
| engineering | Parametric Resynthesis with neural vocoders | https://paperswithcode.com/paper/parametric-resynthesis-with-neural-vocoders |
| materials | Evaluation of GlassNet for physics-informed machine learning of glass stability and glass-forming ability | https://paperswithcode.com/paper/evaluation-of-glassnet-for-physics-informed |
| materials | Predicting materials properties without crystal structure: Deep representation learning from stoichiometry | https://paperswithcode.com/paper/predicting-materials-properties-without |
| mathematics | Deep Learning Evidence for Global Optimality of Gerver's Sofa | https://paperswithcode.com/paper/deep-learning-evidence-for-global-optimality |
| medicine | Drug-Drug Adverse Effect Prediction with Graph Co-Attention | https://paperswithcode.com/paper/drug-drug-adverse-effect-prediction-with |
| medicine | Osteoporosis screening: Leveraging EfficientNet with complete and cropped facial panoramic radiography imaging | https://paperswithcode.com/paper/osteoporosis-screening-leveraging |
| medicine | mmFormer: Multimodal Medical Transformer for Incomplete Multimodal Learning of Brain Tumor Segmentation | https://paperswithcode.com/paper/mmformer-multimodal-medical-transformer-for |
| neuroscience | Conditional Temporal Attention Networks for Neonatal Cortical Surface Reconstruction | https://paperswithcode.com/paper/conditional-temporal-attention-networks-for |
| neuroscience | Estimating Musical Surprisal in Audio | https://paperswithcode.com/paper/estimating-musical-surprisal-in-audio |
| neuroscience | The Developing Human Connectome Project: A Fast Deep Learning-based Pipeline for Neonatal Cortical Surface Reconstruction | https://paperswithcode.com/paper/the-developing-human-connectome-project-a |
| none | Better Patch Stitching for Parametric Surface Reconstruction | https://paperswithcode.com/paper/better-patch-stitching-for-parametric-surface |
| none | GREED: A Neural Framework for Learning Graph Distance Functions | https://paperswithcode.com/paper/a-neural-framework-for-learning-subgraph-and |
| none | Planted Dense Subgraphs in Dense Random Graphs Can Be Recovered using Graph-based Machine Learning | https://paperswithcode.com/paper/planted-dense-subgraphs-in-dense-random |
| physics | Physics-inspired spatiotemporal-graph AI ensemble for the detection of higher order wave mode signals of spinning binary black hole mergers | https://paperswithcode.com/paper/physics-inspired-spatiotemporal-graph-ai |
| physics | Probabilistic neural operators for functional uncertainty quantification | https://paperswithcode.com/paper/probabilistic-neural-operators-for-functional |
| social_science | Rethinking Fair Graph Neural Networks from Re-balancing | https://paperswithcode.com/paper/rethinking-fair-graph-neural-networks-from-re |
