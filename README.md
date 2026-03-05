# RePro: Rectifying LLM Thought From Lens of Optimization

## 📋 Introduction

Recent advancements in large language models (LLMs) have been driven by their
emergent reasoning capabilities, particularly through long chain-of-thought (CoT)
prompting, which enables thorough exploration and deliberation. Despite these
advances, long-CoT LLMs often exhibit suboptimal reasoning behaviors, such
as overthinking and excessively protracted reasoning chains, which can impair
performance. In this paper, we analyze reasoning processes through an optimization
lens, framing CoT as a gradient descent procedure where each reasoning step
constitutes an update toward problem resolution. Building on this perspective,
we introduce RePro (**Re**ctifying **Pro**cess-level Reward), a novel approach to
refine LLM reasoning during post-training. RePro defines a surrogate objective
function to assess the optimization process underlying CoT, utilizing a dual scoring
mechanism to quantify its intensity and stability. These scores are aggregated
into a composite process-level reward, seamlessly integrated into reinforcement
learning with verifiable rewards (RLVR) pipelines to optimize LLMs. Extensive
experiments across multiple reinforcement learning algorithms and diverse LLMs,
evaluated on benchmarks spanning mathematics, science, and coding, demonstrate
that RePro consistently enhances reasoning performance and mitigates suboptimal
reasoning behaviors.


## ⚙️ Installation

```
conda create -n repro python=3.10
conda activate repro
pip install -e .
```

## 🚀 Quick Start

### 🧠 GRPO Trainin

```
scripts/run_multinodes_repro_grpo.sh \
    [MODEL_NAME] \
    [NUM_NODE] \
    [NUM_GPU_PER_NODE] \
    [TP_FOR_ROLLOUT] \
    [GPU_MEMORY_UTLIZATION_FOR_ROLLOUT] \
    deepscale-r-preview
```