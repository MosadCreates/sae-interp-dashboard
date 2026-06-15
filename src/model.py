"""
GPT-2 Small model loader and activation extractor.

Two hook backends:
  - "hf" (default): uses transformers' output_hidden_states for residual_post,
    and PyTorch forward hooks for mlp_post/attn_out/mlp_in
  - "transformer_lens": uses transformer_lens library (optional, requires pip install)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class ActivationExtractor:
    """Extracts hidden-state activations from a HuggingFace GPT-2 model.

    Hook point options:
      - "residual_post" (default): output of the transformer block (residual stream).
        Uses output_hidden_states=True — no forward hooks needed.
      - "mlp_post": output of the MLP sublayer (before residual add).
        Uses a forward hook on block.mlp.
      - "attn_out": output of the attention sublayer (before residual add).
        Uses a forward hook on block.attn.
      - "mlp_in": input to the MLP sublayer.
        Uses a forward hook on block.mlp, capturing the input.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        layer: int = 8,
        hook_point: str = "residual_post",
        device: str = "cpu",
        hook_backend: str = "hf",
    ):
        self.device = torch.device(device)
        self.hook_backend = hook_backend
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,  # always set; used by residual_post
            torch_dtype=torch.float32,
        ).to(self.device)
        self.model.eval()

        self.layer = layer
        self.hook_point = hook_point

        self.captured: List[torch.Tensor] = []
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

        if hook_point == "residual_post":
            # output_hidden_states=True gives us hidden states for free
            pass
        else:
            self._register_hook()

    def _get_target_module(self) -> nn.Module:
        block = self.model.transformer.h[self.layer]
        targets: Dict[str, nn.Module] = {
            "mlp_post": block.mlp,
            "attn_out": block.attn,
            "mlp_in": block.mlp,
        }
        module = targets.get(self.hook_point)
        if module is None:
            raise ValueError(
                f"Unknown hook_point '{self.hook_point}'. "
                f"Options: residual_post, mlp_post, attn_out, mlp_in"
            )
        return module

    def _make_hook(self):
        def hook(module: nn.Module, inp: Tuple[torch.Tensor], out: torch.Tensor):
            if self.hook_point == "mlp_in":
                self.captured.append(inp[0].detach().cpu())
            else:
                self.captured.append(out.detach().cpu())
        return hook

    def _register_hook(self):
        module = self._get_target_module()
        self._hook_handle = module.register_forward_hook(self._make_hook())

    def __call__(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run a forward pass and return captured activations.

        Returns:
            Tensor of shape [batch, seq_len, d_model].
        """
        if self.hook_point == "residual_post":
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=(
                        attention_mask.to(self.device)
                        if attention_mask is not None
                        else None
                    ),
                    output_hidden_states=True,
                )
            # hidden_states[0] = embeddings, hidden_states[k] = block k-1 output
            # For layer L, the residual stream output is hidden_states[L+1]
            return outputs.hidden_states[self.layer + 1].detach().cpu()
        else:
            self.captured = []
            with torch.no_grad():
                self.model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=(
                        attention_mask.to(self.device)
                        if attention_mask is not None
                        else None
                    ),
                )
            if not self.captured:
                raise RuntimeError("No activations captured — check hook registration.")
            return self.captured[0]

    def clean(self) -> None:
        self.captured = []

    def remove_hooks(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
