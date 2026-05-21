# **Layer-wise Candidate-conditioned Separability and Internal Robustness in Large Language Models**

Research-oriented framework for analysing layer-wise candidate-conditioned separability and internal robustness in transformer-based large language models (LLMs).

This repository contains the source code for my master's thesis project:

> **Layer-wise Candidate-conditioned Separability and Internal Robustness in Large Language Models**  
> **Nasim Maleki Najafabadi — Dalarna University**

---

# **Research Focus**

This project studies two main questions:

## **RQ1 — Candidate-conditioned separability**

At what depth do correct and incorrect candidate-conditioned completions become linearly separable in hidden-state representations?

## **RQ2 — Internal robustness**

How sensitive are model outputs and candidate-conditioned separability signals to perturbations of hidden activations?

---

# **Evaluated Models**

The experiments use two open-weight transformer-based language models:

- **Qwen2.5-0.5B**
- **DeepSeek-R1-Distill-Qwen-1.5B**

Both models are evaluated in a frozen setting without fine-tuning.

---

# **Experimental Pipeline**

```text
Prompt generation
        ↓
Model-specific tokenisation filtering
        ↓
Candidate-conditioned sequence construction
        ↓
Layer-wise hidden-state extraction
        ↓
Linear probing
        ↓
Probe-based EDL-window detection
        ↓
Internal robustness analysis
        ↓
Result summarisation and plotting
```

---

# **Repository Structure**

```text
master_thesis/
│
├── scripts/
│   ├── make_prompt_pool.py
│   ├── extract_features.py
│   ├── run_linear_probes.py
│   ├── e2e_robustnes.py
│   ├── run_local_logitlens_robustness.py
│   ├── run_probe_space_robustness.py
│   ├── make_edl_and_plots_all_methods.py
│   ├── edl_threshold_sensitivity.py
│   ├── overfit_generalization_gap_per_model.py
│   ├── summarize_prompt_pools.py
│   ├── summarize_e2e_auc.py
│   ├── summarize_e2e_table.py
│   ├── summarize_noise_report.py
│   ├── summarize_probe_table11.py
│   ├── plot_e2e_across_models.py
│   ├── plot_logitlens_robustness_across_models.py
│   ├── plot_probe_robustness_across_models.py
│   └── probes_summary.py
│
├── src/
│   ├── feature_cache.py
│   ├── hooks.py
│   ├── io.py
│   ├── linear_probes.py
│   ├── logit_lens.py
│   ├── mass_mean.py
│   ├── metrics.py
│   ├── probing.py
│   ├── tuned.py
│   ├── util.py
│   └── viz.py
│
├── run_make_prompts.sh
├── run_extract_features.sh
├── run_linear_probes_by_split.sh
├── run_linear_probes_rep.sh
├── run_linear_probes_rep1.sh
├── run_e2e_robustnes_pipeline.sh
├── run_local_robustnes_logit_pipeline.sh
├── run_local_run_probe_space_robustness_pipeline.sh
├── run_edl_plots.sh
├── run_edl_threshold_sensitivity.sh
├── run_overfit_generalization_gap_per_model.sh
├── run_dataset_summary.sh
│
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

---

# **Main Components**

## **Prompt generation**

Controlled prompt pools are generated for:

- arithmetic MCQ tasks
- arithmetic Single-token tasks
- capital-city True/False MCQ tasks
- capital-city True/False Single-token tasks

The framework applies model-specific tokenisation checks to enforce single-token constraints.

---

## **Hidden-state extraction**

The framework extracts:

- layer-wise hidden-state representations
- candidate-conditioned representations
- last-token hidden states

These representations are cached and reused in later experiments.

---

## **Linear probing**

Layer-wise probing experiments use:

- **Logistic Regression**
- **Linear SVM**

The evaluation includes:

- AUROC
- question-level accuracy
- mean margin analysis

---

## **Probe-based EDL windows**

The repository includes:

- layer-wise AUROC tracking
- threshold-based EDL detection
- sustained separability analysis

---

## **Robustness analysis**

The framework evaluates:

- end-to-end robustness
- local Logit Lens robustness
- probe-space robustness
- Gaussian noise perturbations
- layer-wise robustness profiles

---

# **Key Features**

- candidate-conditioned hidden-state analysis
- layer-wise probing pipeline
- robustness evaluation under internal perturbations
- EDL-window analysis
- Logit Lens robustness analysis
- probe-space robustness analysis
- automated experiment pipelines
- reproducible research-oriented framework

---

# **Technologies Used**

- **Python**
- **PyTorch**
- **Hugging Face Transformers**
- **scikit-learn**
- **NumPy**
- **Pandas**
- **Matplotlib**

---

# **Data and Generated Outputs**

Large generated artefacts are intentionally excluded from this repository.

The following files and directories are not included in Git:

- generated prompt pools
- cached hidden-state features
- model outputs
- robustness outputs
- generated plots
- intermediate experiment results
- downloaded model files

These artefacts can be regenerated locally by running the experiment scripts.

This design keeps the repository lightweight and focused on the source code and reproducible experimental pipeline.

---

# **Installation**

Clone the repository:

```bash
git clone https://github.com/malekinasim/master_thesis.git
cd master_thesis
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

## **Linux / macOS**

```bash
source .venv/bin/activate
```

## **Windows**

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# **Example Usage**

## **1. Generate prompt pools**

```bash
bash run_make_prompts.sh
```

## **2. Extract hidden-state features**

```bash
bash run_extract_features.sh
```

## **3. Run linear probing**

```bash
bash run_linear_probes_by_split.sh
```

## **4. Run end-to-end robustness analysis**

```bash
bash run_e2e_robustnes_pipeline.sh
```

## **5. Run local Logit Lens robustness**

```bash
bash run_local_robustnes_logit_pipeline.sh
```

## **6. Run probe-space robustness**

```bash
bash run_local_run_probe_space_robustness_pipeline.sh
```

## **7. Generate EDL plots**

```bash
bash run_edl_plots.sh
```

---

# **Key Analyses**

This repository supports:

- layer-wise hidden-state extraction
- candidate-conditioned separability analysis
- Logistic Regression and Linear SVM probing
- AUROC and accuracy curves across transformer depth
- probe-based Early Decision Layer window detection
- end-to-end robustness under internal noise injection
- local Logit Lens robustness
- probe-space robustness
- margin-vs-robustness analysis
- overfitting and generalisation-gap diagnostics

---

# **Key Findings**

The experiments show that:

- correct and incorrect candidate-conditioned completions become increasingly linearly separable across transformer depth;
- strong separability often appears before the final transformer layer;
- robustness is strongly dependent on the evaluation pathway;
- probe-space robustness can remain high even when local Logit Lens decoding is fragile;
- candidate-conditioned separability and robustness are related but not equivalent.

Importantly, the probing analysis is based on candidate-conditioned sequences. Therefore, the results should not be interpreted as evidence of prompt-only answer formation.

---

# **Academic Context**

This repository accompanies a master's thesis submitted at:

**Dalarna University**  
**Master’s Degree in Microdata Analysis**

**Supervisor:** Arend Hintze  
**Examiner:** Mia Xiaoyun Zhao

---

# **Citation**

```bibtex
@mastersthesis{maleki2026,
  title={Layer-wise Candidate-conditioned Separability and Internal Robustness in Large Language Models},
  author={Maleki Najafabadi, Nasim},
  school={Dalarna University},
  year={2026}
}
```

---

# **Author**

**Nasim Maleki Najafabadi**

Research interests:

- LLM interpretability
- transformer internal representations
- candidate-conditioned separability
- robustness analysis
- machine learning systems

GitHub: https://github.com/malekinasim/master_thesis

---

# **License**

This repository is provided for academic and research purposes.
