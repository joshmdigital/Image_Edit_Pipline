# Natural Language Image Editing Pipeline

A four-phase, language-driven image editing system built end-to-end on **Google Colab**. Given a source image and a plain-English instruction (e.g. *"Change the color of the dress to a soft pink"*), the pipeline localizes the target region and renders the edit using a chained VLM → grounding → segmentation → diffusion stack.

> **This repo is an illustration, not a SOTA submission.** The point is to show that a non-trivial multimodal pipeline — fine-tuning a VLM, training a diffusion bridge, and assembling a four-stage inference graph — can be built, trained, and evaluated entirely within the constraints of a free-tier-adjacent cloud environment (Colab Pro A100 + Google Drive). Every architectural and data choice in this project is shaped by those constraints. See the [full report PDF](./ImgEdit%20Report.pdf) for the complete write-up, results, and discussion.

---

## What it does

Input: a source image + a natural-language edit instruction.
Output: the same image with the requested edit applied to the correct region.

Example: *"Change the color of the dress to a soft pink"* → the dress in the image is repainted soft pink, the rest of the scene is preserved.

## Architecture

The system chains four trained or pretrained components at inference time:

| Stage | Component | Role |
|---|---|---|
| 1 | **Qwen2.5-VL-3B-Instruct** (QLoRA fine-tuned) | Parse the instruction against the image; emit structured JSON (`edit_type`, `edit_description`, subject phrase) and a 2048-dim hidden-state conditioning vector |
| 2 | **GroundingDINO** (`grounding-dino-tiny`, zero-shot) | Localize the target object from the subject phrase → bounding box |
| 3 | **SAM2** (`sam2.1_hiera_large`, frozen) | Convert the bounding box into a precise binary mask |
| 4 | **Stable Diffusion 1.5 Inpainting** (UNet LoRA + MLP projection adapter, trained) | Inpaint the masked region, conditioned on CLIP text embeddings concatenated with the projected VLM hidden state |

Two components are trained in this project: the Phase 1 VLM (QLoRA on attention projections) and the Phase 2 diffusion bridge (LoRA on UNet attention + a 2-layer MLP that maps the 2048-dim VLM hidden state into SD's 768-dim cross-attention space). Everything else is used off-the-shelf.

## Phase breakdown

Each phase is a self-contained Colab notebook. Artifacts are persisted to Google Drive between phases so a phase can be re-run without re-running the previous one.

- **Phase 0 — Data.** Stream the `sysuyy/ImgEdit` dataset from HuggingFace, join with the companion `sysuyy/ImgEdit_recap_mask` dataset for ground-truth RLE masks, filter for quality. Final training set: 7,818 `adjust`-type pairs.
- **Phase 1 — VLM fine-tuning.** QLoRA fine-tune Qwen2.5-VL-3B with TRL `SFTTrainer` to emit structured JSON edit predictions. After training, cache mean-pooled last-layer hidden states for every training sample to Drive.
- **Phase 2 — Diffusion bridge training.** Train the `VLMProjectionAdapter` MLP and UNet LoRA adapters jointly on the pre-cached hidden states + ground-truth RLE masks. Standard epsilon-prediction MSE objective.
- **Phase 3 — Inference pipeline.** Chain the four components above end-to-end.
- **Phase 4 — Benchmark evaluation.** Run the full pipeline on the 67 `adjust`-type samples from the ImgEdit singleturn benchmark and visually inspect outputs.

## Why "illustration" and not a benchmark submission

The design space was bounded throughout by what a single A100 with a 24-hour wall-clock and a Google Drive quota can actually do:

- Training corpus restricted to a single edit type (`adjust`), 7,818 samples, because the `.tar.split` archive layout makes a stratified cross-type sample prohibitively slow to download.
- VLM hidden states pre-cached to Drive rather than computed live during diffusion training — keeps the 3B VLM and the SD UNet from needing to be co-resident in VRAM.
- Inference resolution fixed at 512×512 (SD 1.5 native).
- An open-vocabulary detector (GroundingDINO) was added to Phase 3 specifically because the fine-tuned VLM's bounding-box predictions, while accurate on training images, were too noisy on unseen benchmark images to drive SAM2 directly. A higher training budget on a larger VLM variant would likely make this step removable.

The report discusses each of these trade-offs in detail, including a future-work section on a more efficient Parquet-driven sampling strategy that would scale to ~24k–50k diverse training pairs within the same platform.

## Results Summary 

All 67 benchmark samples completed inference (100% completion rate, ~8.4s per sample on A100). The pipeline performs well on edits to spatially compact, visually distinct objects with simple silhouettes and a simple color-change instruction (e.g. dress → soft pink, yacht hull → navy blue). It degrades on complex geometry (motorcycles, seaplane propellers, building facades), full-surface repaints where the model must synthesize rather than recolor (animal fur), and instructions where the current and target color are both named lexically (the model anchored on the source color). Most failure modes trace back to training data scale and diversity rather than architectural defects. Detailed per-sample analysis and figures are in [the report](./ImgEdit_Report.pdf).


## Running it yourself

The notebooks assume Colab Pro with an A100 runtime and Google Drive mounted at `/content/drive/MyDrive/img_edit_pipeline/`. Each notebook starts with a pinned-version `pip install` cell — those pins matter (Qwen2.5-VL's Transformers integration, PEFT's LoRA API, and SAM2's PyTorch requirement are all version-sensitive). See Table 1 in the report for the full dependency matrix per phase.

You will need:

- A HuggingFace account with access to `sysuyy/ImgEdit` and `sysuyy/ImgEdit_recap_mask`.
- The SAM2 checkpoint (`sam2.1_hiera_large.pt`) cached to Drive; SAM2 itself is installed from its GitHub source rather than PyPI.
- Roughly 50 GB of Drive headroom for the filtered training set, cached hidden states, and model checkpoints.

Phases are independently re-runnable as long as the previous phase's artifacts are present in Drive.

## Reference

For methodology, dataset details, per-sample qualitative analysis, the discussion of limitations, and the future-work proposal for a Parquet-driven stratified sampling strategy, see:

**[ImgEdit_Report.pdf](./ImgEdit%20Report.pdf)** — *Image Editing Vision Language Model Pipeline Project*, Joshua Moffat, SDSU CS 668 (Applied LLMs), May 2026.

## Key references

- ImgEdit dataset and benchmark — Ye et al., 2025 ([arXiv:2505.20275](https://arxiv.org/abs/2505.20275))
- Qwen2.5-VL — Wang et al., 2024 ([arXiv:2409.12191](https://arxiv.org/abs/2409.12191))
- QLoRA — Dettmers et al., 2023 ([arXiv:2305.14314](https://arxiv.org/abs/2305.14314))
- GroundingDINO — Liu et al., 2024 ([arXiv:2303.05499](https://arxiv.org/abs/2303.05499))
- SAM 2 — Ravi et al., 2024 ([arXiv:2408.00714](https://arxiv.org/abs/2408.00714))
- Stable Diffusion / Latent Diffusion Models — Rombach et al., 2022 (CVPR)

## License

The code in this repository (notebooks, scripts, configuration) is released under the **Apache License 2.0** — see [LICENSE](./LICENSE). This repo contains no model weights, trained adapters, or dataset samples; only the code to produce and use them.

### Third-party components

The pipeline loads or fine-tunes the following components at runtime. Each has its own license that governs your use of that component:

| Component | Source | License |
|---|---|---|
| Qwen2.5-VL-3B-Instruct | [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Qwen Research License (research / non-commercial) |
| Stable Diffusion 1.5 Inpainting | [`runwayml/stable-diffusion-inpainting`](https://huggingface.co/runwayml/stable-diffusion-inpainting) | CreativeML OpenRAIL-M |
| SAM 2 | [`facebookresearch/sam2`](https://github.com/facebookresearch/sam2) | Apache 2.0 |
| GroundingDINO (`grounding-dino-tiny`) | [`IDEA-Research/grounding-dino-tiny`](https://huggingface.co/IDEA-Research/grounding-dino-tiny) | Apache 2.0 |
| ImgEdit dataset | [`sysuyy/ImgEdit`](https://huggingface.co/datasets/sysuyy/ImgEdit), [`sysuyy/ImgEdit_recap_mask`](https://huggingface.co/datasets/sysuyy/ImgEdit_recap_mask) | See dataset cards |

Because Qwen2.5-VL-3B is research-licensed, the assembled pipeline as a whole is not suitable for commercial deployment without substituting a commercially-licensed VLM. Any adapter weights produced by running this code are derivatives of their base models and inherit those licenses; label them accordingly if you publish them.
