# Transparent, Domain-Adapted Detection of Adverse Drug Reactions from Patient Reviews

**With a faithful explanation layer**

MSc in Artificial Intelligence research project · National College of Ireland, School of Computing

![Python](https://img.shields.io/badge/python-3.x-blue) ![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow) ![Docker](https://img.shields.io/badge/deploy-Docker-2496ED) ![FastAPI](https://img.shields.io/badge/API-FastAPI-009688) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)

---

## Overview

Adverse drug reactions (ADRs) are a major source of preventable harm, yet only a small fraction reach spontaneous pharmacovigilance reporting systems. Patient-written drug reviews are a large, timely, real-world safety signal, but they are underused for automated detection because of two barriers:

1. **Domain mismatch.** Models trained on curated biomedical text transfer poorly to informal, lay-language patient text.
2. **Opacity.** Predictions come with no established, non-hallucinated rationale, so flagged ADRs cannot earn clinical trust.

This project builds a **domain-aware ADR detector** that classifies whether a sentence contains an ADR, extracts the exact ADR spans, and attaches a **verifiable, extractive explanation** to every prediction. Explanation faithfulness is measured rather than assumed.

> **Research question:** Can a domain-adapted model detect ADRs in patient reviews and accompany every prediction with a faithful, verifiable explanation, across different registers and corpora?

## Key findings

| Finding | Result |
|---|---|
| In-domain classification (ADE-Corpus-V2, fine-tuned PubMedBERT) | F1 **0.954**, AUC **0.992** (3-seed F1 0.934 ± 0.009) |
| In-domain classification (soft-voting ensemble) | F1 0.949, AUC 0.994 |
| In-domain NER (ADR spans) | Strict F1 **0.844**, partial F1 **0.943** |
| **Zero-shot on patient reviews** (negative result) | AUC **0.58**, F1 0.04, near chance |
| After domain adaptation on CADEC | AUC **0.97**, F1 0.91 |
| Single combined checkpoint | Reviews AUC 0.97, abstracts AUC 0.99, with light forgetting |
| Cross-corpus transfer to SMM4H tweets | Classifier: F1 0.69 / AUC 0.86. NER span extraction does **not** transfer |
| Explanation grounding | Extractive rationale flagged ungrounded in **0%** of cases vs **90%** for a generative T5 baseline (McNemar χ² ≈ 52, p < 0.001, 60 paired examples) |

**Takeaways**

- In-domain benchmarks substantially overstate performance on patient reviews.
- Domain adaptation closes the gap for sentence classification.
- Adaptation is asymmetric: classification transfers moderately across review sources, while span extraction stays corpus-specific.

## System architecture

The system has two coordinated layers.

**ADR detection pipeline**

1. **Ingest and preprocess:** length filter, deduplication, RxNorm drug-name normalisation.
2. **Classify (ensemble):** PubMedBERT + SciBERT, with probabilities averaged (soft voting) before thresholding.
3. **Recognise (NER):** a separate PubMedBERT with a 3-class BIO head (`O` / `B-ADR` / `I-ADR`).

**XAI and evaluation layer**

1. **Explain:** attention rollout and Integrated Gradients.
2. **Ground:** an extractive rationale that quotes NER spans and top tokens verbatim from the review, checked by a hallucination guard.
3. **Serve:** FastAPI + Streamlit, Dockerised.

### Hallucination guard

Every rationale is independently re-scored on:

- Token overlap with the source review
- Literal span coverage of quoted text
- NLI entailment (review ⊨ rationale). This is implemented as an optional signal.

The reported faithfulness figures use the two literal-grounding signals (token overlap and span coverage).

Because the served rationale only quotes text from the review, an unsupported claim is structurally impossible.

## Method

| Stage | What happens |
|---|---|
| 1. Collect | Four corpora spanning abstracts, reviews, forums and tweets |
| 2. Clean and rebuild | Fix a label defect in the ADE setup; rebuild BIO tags from character offsets with leakage-free splitting |
| 3. Train and adapt | Fine-tune in-domain, then domain-adapt to patient reviews |
| 4. Explain | Build the explanation layer |
| 5. Evaluate | In-domain, review-domain and cross-corpus tests, plus explanation grounding |

**Weak supervision.** Self-training on UCI reviews, filtered with NegEx (negation) and an indication filter so that, for example, "Valium for anxiety" is not tagged as a reaction. About 3k pseudo-labelled rows are added to the ADE gold set.

**Domain adaptation.** Further fine-tuning on CADEC patient-forum posts to close the register gap.

**Training setup.** HuggingFace `Trainer`, AdamW (lr 2e-5), fp16, early stopping, repeated over seeds 42 / 123 / 456. The generative baseline path uses Flan-T5-small.

## Datasets

| Corpus | Register | Role |
|---|---|---|
| ADE-Corpus-V2 | Biomedical abstracts | Gold training and in-domain evaluation |
| UCI Drug Reviews (~215K) | Consumer reviews | Weak-labelled NER augmentation |
| CADEC | Patient-forum posts | Domain adaptation and review-domain evaluation |
| SMM4H | Twitter posts | Held-out cross-corpus generalisation test |

> **Data are not redistributed in this repository.** Please obtain each dataset from its original source and check its licence and usage terms.

## Fairness audit

A fairness audit across demographic-proxy subgroups found a maximum F1 disparity of 0.111. The largest gap is acute vs. chronic recall (0.82 vs. 0.96, Δ 0.147), which is the priority fix before any clinical use.

## Limitations

- Initial training on a single biomedical corpus drives the domain gap.
- Faithfulness is an automated proxy, not yet validated by a human pharmacist study.
- Some figures rely on single-seed runs and modest review test sets.
- Span extraction remains corpus-specific across social-media text.
- This is a research prototype and **not a clinical tool**. It must not be used for medical decisions.

## Repository structure

<!-- TODO: replace with your actual layout -->
```
.
├── data/            # dataset instructions / loaders (no raw data)
├── src/             # preprocessing, training, explanation, evaluation
├── app/             # FastAPI service and Streamlit UI
├── notebooks/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Getting started

<!-- TODO: replace the commands below with your real entry points -->

```bash
# 1. Clone
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Install
pip install -r requirements.txt

# 3. Reproduce the pipeline
# python <your_pipeline_script>.py

# 4. Run the demo with Docker
docker build -t adr-detector .
docker run -p 8501:8501 adr-detector
```

## Future work

- Expert pharmacist annotation study for clinically grounded agreement
- Knowledge-graph layer linking ADR spans to drug-reaction pathways
- Multi-corpus training to shrink the extraction gap
- CRF or span-based heads, and closing the acute-vs-chronic disparity

## Key references

- Gu et al. (2021). Domain-specific language model pretraining for biomedical NLP (PubMedBERT). *ACM Trans. Computing for Healthcare.*
- Beltagy et al. (2019). SciBERT: A pretrained language model for scientific text. *EMNLP-IJCNLP.*
- Lee et al. (2020). BioBERT: a pre-trained biomedical language representation model. *Bioinformatics.*
- Portelli et al. (2021). A comparison of transformer architectures for ADE detection. *EACL.*
- Scaboro et al. (2023). Extensive evaluation of transformer-based architectures for ADE extraction. *Knowledge-Based Systems.*
- Nikfarjam et al. (2015). Pharmacovigilance from social media: sequence labeling with word-embedding features. *JAMIA.*
- Karimi et al. (2015). CADEC: A corpus of adverse drug event annotations. *J. Biomedical Informatics.*
- Gurulingappa et al. (2012). Benchmark corpus for drug-related adverse effects (ADE-Corpus-V2). *J. Biomedical Informatics.*
- Jacovi & Goldberg (2020). Towards faithfully interpretable NLP systems: how should we define and evaluate faithfulness? *ACL.*
- Sundararajan et al. (2017). Axiomatic attribution for deep networks (Integrated Gradients). *ICML.*

The full reference list (19 sources) is in the accompanying report.

## Author

**Plabon Banik**
MSc Artificial Intelligence, National College of Ireland
GitHub: [Turjoy167](https://github.com/Turjoy167)

## Citation

<!-- TODO: update once the thesis/paper has a final title, year and link -->
```bibtex
@mastersthesis{banik2026adr,
  author = {Banik, Plabon},
  title  = {Transparent, Domain-Adapted Detection of Adverse Drug Reactions from Patient Reviews with a Faithful Explanation Layer},
  school = {National College of Ireland},
  year   = {2026}
}
```

## License

<!-- TODO: choose a licence (e.g. MIT) and add a LICENSE file. Dataset licences are separate. -->
