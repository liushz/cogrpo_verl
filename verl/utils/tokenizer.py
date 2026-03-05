# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for tokenization."""

import warnings

__all__ = ["hf_tokenizer", "hf_processor"]


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def _patch_transformers_update_trie():
    """Compatibility patch for some remote tokenizers.

    Some tokenizer implementations (e.g. InternS1Tokenizer) call `_update_trie()` during
    `add_tokens()` initialization, but recent `transformers` versions (>=v5) may not
    expose this method on `PreTrainedTokenizerBase` for fast tokenizers.

    We provide a minimal implementation that syncs `_extra_special_tokens` into the
    Rust backend when present.
    """
    try:
        from transformers.tokenization_utils_base import AddedToken, PreTrainedTokenizerBase
    except Exception:
        return

    if not hasattr(PreTrainedTokenizerBase, "_update_trie"):
        def _update_trie(self, unique_no_split_tokens=None):  # noqa: ARG001
            try:
                backend = getattr(self, "_tokenizer", None)
                if backend is None:
                    return

                extra = getattr(self, "_extra_special_tokens", None)
                if not extra:
                    return

                to_add = []
                for tok in extra:
                    if isinstance(tok, AddedToken):
                        content = tok.content
                        if backend.token_to_id(content) is None:
                            to_add.append(tok)
                    else:
                        content = str(tok)
                        if backend.token_to_id(content) is None:
                            to_add.append(AddedToken(content, special=True, normalized=False))

                if not to_add:
                    return

                try:
                    backend.add_special_tokens(to_add)
                except Exception:
                    backend.add_tokens(to_add)
            except Exception:
                # Best-effort patch: never fail tokenizer init because of this shim.
                return

        PreTrainedTokenizerBase._update_trie = _update_trie  # type: ignore[attr-defined]

    if not hasattr(PreTrainedTokenizerBase, "_update_total_vocab_size"):
        def _update_total_vocab_size(self):
            # Some v4-era tokenizers expect this helper. In v5 fast tokenizers, `len(self)`
            # already reflects the Rust backend size, so we keep a best-effort mirror field.
            try:
                total_vocab_size = len(self.get_vocab())
            except Exception:
                try:
                    total_vocab_size = len(self)
                except Exception:
                    return

            try:
                setattr(self, "total_vocab_size", int(total_vocab_size))
            except Exception:
                return

        PreTrainedTokenizerBase._update_total_vocab_size = _update_total_vocab_size  # type: ignore[attr-defined]


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer

    _patch_transformers_update_trie()

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn("Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1)
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except Exception as e:
        processor = None
        # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid
        # silent failure
        warnings.warn(f"Failed to create processor: {e}. This may affect multimodal processing", stacklevel=1)
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor
