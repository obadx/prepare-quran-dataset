"""Debug script: trace activation explosion across conformer layers.

Measures per-submodule magnitude at every point in every encoder layer,
with detailed pre/post activation values inside FFN and Conv blocks.
Prints a compact waterfall table + amplification factors.
"""

import json
import math
import torch
import torch.nn as nn
import numpy as np

from transformers import AutoFeatureExtractor
from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import (
    Wav2Vec2BertEncoderLayer,
    Wav2Vec2BertFeedForward,
)

from prepare_quran_dataset.modeling_streaming_rnn.configuration_rnn_streaming_multi_level_ctc import (
    Wav2Vec2BertForRNNStreamingMultilevelCTCConfig,
)
from prepare_quran_dataset.modeling_streaming_rnn.modeling_rnn_streaming_multi_level_ctc import (
    Wav2Vec2BertForRNNStreamingMultilevelCTC,
)

with open("./vocab.json", encoding="utf-8") as f:
    _vocab = json.load(f)
_level_to_vocab_size = {l: len(v) for l, v in _vocab.items()}


def fmt(x, width=8):
    """Format a scalar for table output."""
    if isinstance(x, bool):
        return "  NaN! " if x else "       "
    if not np.isfinite(x):
        return (
            "   Inf "
            if x == float("inf")
            else "  -Inf "
            if x == float("-inf")
            else "   NaN "
        )
    if abs(x) < 0.001:
        return f"{x:.2e}".rjust(width)
    if abs(x) < 1000:
        return f"{x:>{width}.3f}"
    return f"{x:>{width}.1f}"


def stats_dict(t, label):
    return {
        "label": label,
        "max_abs": t.abs().max().item(),
        "mean": t.mean().item(),
        "std": t.std().item(),
        "max": t.max().item(),
        "min": t.min().item(),
        "has_inf": torch.isinf(t).any().item(),
        "has_nan": torch.isnan(t).any().item(),
    }


# ── measurement registry ──────────────────────────────────────────────────────


class Measure:
    """Collects stats at named measurement points during forward."""

    def __init__(self):
        self.records = {}  # (layer_idx, step_name) → stats_dict
        self.layer_order = []
        self._current_layer = -1

    def set_layer(self, idx):
        self._current_layer = idx
        if idx not in self.layer_order:
            self.layer_order.append(idx)

    def record(self, t, step_name):
        key = (self._current_layer, step_name)
        self.records[key] = stats_dict(t, step_name)

    def amplification(self, key_in, key_out):
        """σ(out) / σ(in) for a submodule."""
        if key_in not in self.records or key_out not in self.records:
            return None
        s_in = self.records[key_in]["std"]
        s_out = self.records[key_out]["std"]
        if s_in == 0:
            return float("inf")
        return s_out / s_in

    def table_rows(self, layers=None):
        """Yield formatted rows for display."""
        if layers is None:
            layers = self.layer_order
        step_names = [
            "input",
            "ffn1_ln_in",
            "ffn1_ln_out",
            "ffn1_pre_act",
            "ffn1_post_inter",
            "ffn1_out",
            "ffn1_res",
            "attn_ln_in",
            "attn_ln_out",
            "attn_Q_proj",
            "attn_K_proj",
            "attn_V_proj",
            "attn_Q_head",
            "attn_K_head",
            "attn_Q_per_head",
            "attn_scores_raw",
            "attn_scores",
            "attn_dist_raw",
            "attn_scores_dist",
            "attn_scores_masked",
            "attn_probs",
            "attn_context",
            "attn_out_internal",
            "attn_out",
            "attn_res",
            "conv_ln_in",
            "conv_ln_out",
            "conv_pre_glu",
            "conv_post_glu",
            "conv_post_depth",
            "conv_post_act",
            "conv_out",
            "conv_res",
            "ffn2_ln_in",
            "ffn2_ln_out",
            "ffn2_pre_act",
            "ffn2_post_inter",
            "ffn2_out",
            "ffn2_res",
            "final_ln",
        ]
        header = f"{'Layer':>5} | {'Step':<18} | {'max_abs':>9} | {'mean':>9} | {'std':>9} | {'has_inf':>7} | {'has_nan':>7} | {'max':>9} | {'min':>9}"
        yield header
        yield "-" * len(header)
        previous = None
        for layer in layers:
            for sn in step_names:
                key = (layer, sn)
                if key in self.records:
                    r = self.records[key]
                    # Mark changes
                    inf_mark = "  ↑Inf " if r["has_inf"] else ""
                    nan_mark = "  ◆NaN " if r["has_nan"] else ""
                    if (
                        previous is not None
                        and r["has_nan"]
                        and not previous["has_nan"]
                    ):
                        nan_mark = " ◄◆FIRST NaN"
                    previous = r
                    yield (
                        f"{layer:>5} | {sn:<18} | {fmt(r['max_abs'], 9)} | {fmt(r['mean'], 9)} "
                        f"| {fmt(r['std'], 9)} | {fmt(r['has_inf'], 7)} | {fmt(r['has_nan'], 7)} "
                        f"| {fmt(r['max'], 9)} | {fmt(r['min'], 9)}"
                        f"{inf_mark}{nan_mark}"
                    )

    def amplification_table(self, layers=None):
        if layers is None:
            layers = self.layer_order
        submodules = [
            ("ffn1", "ffn1_ln_out", "ffn1_out"),
            ("attn", "attn_ln_out", "attn_out"),
            ("conv", "conv_ln_out", "conv_out"),
            ("ffn2", "ffn2_ln_out", "ffn2_out"),
        ]
        header = f"{'Layer':>5} | {'ffn1_amp':>9} | {'attn_amp':>9} | {'conv_amp':>9} | {'ffn2_amp':>9} | ffn1_max_abs | conv_max_abs"
        yield header
        yield "-" * len(header)
        for layer in layers:
            amps = []
            for name, k_in, k_out in submodules:
                amp = self.amplification((layer, k_in), (layer, k_out))
                amps.append(fmt(amp, 9) if amp is not None else "     ?   ")
            ffn1_max = self.records.get((layer, "ffn1_out"), {}).get("max_abs", 0)
            conv_max = self.records.get((layer, "conv_out"), {}).get("max_abs", 0)
            yield f"{layer:>5} | {' | '.join(amps)} | {fmt(ffn1_max, 9)} | {fmt(conv_max, 9)}"


# ── attention internal debug ──────────────────────────────────────────────────


def _dump_attn_nan(measure, layer, Q, K, scores_raw, scores, probs, scores_dist=None):
    """Detailed NaN dump for attention internals on first occurrence."""
    if measure._nan_dumped:
        return

    # Check each tensor in order: scores_raw → scores → scores_dist → probs
    candidates = [("scores_raw", scores_raw), ("scores", scores)]
    if scores_dist is not None:
        candidates.append(("scores_dist", scores_dist))
    candidates.append(("probs", probs))

    for name, t in candidates:
        if torch.isnan(t).any().item() or torch.isinf(t).any().item():
            measure._nan_dumped = True
            bsz, heads, seq_len, _ = t.shape

            # Find the most extreme position
            has_nan = torch.isnan(t)
            has_inf = torch.isinf(t)

            print(f"\n  🔥  ATTENTION NaN DETAIL (layer={layer})  🔥")
            print(f"  {'─' * 60}")
            print(f"  NaN/Inf first detected in: {name}")
            print(f"  Sequence length: {seq_len}")

            # ── Find where NaN/Inf is ──
            if has_nan.any():
                nan_positions = torch.where(has_nan)
                sample_idx = 0
                b0 = nan_positions[0][sample_idx].item()
                h0 = nan_positions[1][sample_idx].item()
                r0 = nan_positions[2][sample_idx].item()
                c0 = nan_positions[3][sample_idx].item()
                print(f"  First NaN at:  batch={b0}, head={h0}, row={r0}, col={c0}")
            elif has_inf.any():
                inf_positions = torch.where(has_inf)
                sample_idx = 0
                b0 = inf_positions[0][sample_idx].item()
                h0 = inf_positions[1][sample_idx].item()
                r0 = inf_positions[2][sample_idx].item()
                c0 = inf_positions[3][sample_idx].item()
                print(f"  First Inf at:  batch={b0}, head={h0}, row={r0}, col={c0}")
            else:
                b0, h0, r0, c0 = 0, 0, 0, 0
                print(f"  (no NaN/Inf found in {name})")

            # ── Q and K at the culprit position ──
            q_row = Q[b0, h0, r0, :]
            k_col = K[b0, h0, c0, :]
            q_top_vals, q_top_idx = q_row.abs().topk(min(5, q_row.size(0)))
            k_top_vals, k_top_idx = k_col.abs().topk(min(5, k_col.size(0)))

            print(f"\n  Q[head={h0}, pos={r0}] top-5 dims by magnitude:")
            for i in range(len(q_top_idx)):
                idx = q_top_idx[i].item()
                print(f"    dim={idx:>3}:  val={q_row[idx]:.4f}")

            print(f"\n  K[head={h0}, pos={c0}] top-5 dims by magnitude:")
            for i in range(len(k_top_idx)):
                idx = k_top_idx[i].item()
                print(f"    dim={idx:>3}:  val={k_col[idx]:.4f}")

            # ── Per-head stats before NaN hit ──
            print(f"\n  Per-head diagnostics:")
            for hh in range(min(heads, 16)):
                s = scores_raw[b0, hh]  # use raw scores (before scaling)
                qh = Q[b0, hh]
                kh = K[b0, hh]
                q_max = qh.abs().max().item()
                k_max = kh.abs().max().item()
                s_max = s.abs().max().item()
                s_mean = s.mean().item()
                s_has_nan = torch.isnan(s).any().item()
                s_has_inf = torch.isinf(s).any().item()
                marker = ""
                if s_has_nan:
                    marker = "  ← NaN"
                elif s_has_inf:
                    marker = "  ← Inf"
                if marker:
                    print(
                        f"    head={hh:>2}:  Q_max={q_max:.2f}  K_max={k_max:.2f}  "
                        f"scores_raw_max={s_max:.2f}  scores_raw_mean={s_mean:.4f}{marker}"
                    )

            # ── Find the row with the largest Q norm ──
            q_norms = Q[b0, :, :, :].norm(dim=-1)  # (heads, seq)
            max_q_row = q_norms.argmax(dim=-1)  # (heads,) — best row per head
            print(f"\n  Row with max Q norm per head:")
            for hh in range(heads):
                mr = max_q_row[hh].item()
                print(
                    f"    head={hh:>2}:  row={mr:>3}  Q_norm={q_norms[hh, mr].item():.2f}"
                )

            # ── Raw QK dot product extremes ──
            s_finite = scores_raw[b0, h0][
                ~torch.isnan(scores_raw[b0, h0]) & ~torch.isinf(scores_raw[b0, h0])
            ]
            if s_finite.numel() > 0:
                print(
                    f"\n  Head={h0} scores_raw (finite):  max={s_finite.max().item():.2f}  "
                    f"min={s_finite.min().item():.2f}  mean={s_finite.mean().item():.4f}"
                )

            # Show the max QK product position
            s_finite_all = scores_raw[b0][
                ~torch.isnan(scores_raw[b0]) & ~torch.isinf(scores_raw[b0])
            ]
            if s_finite_all.numel() > 0:
                max_val = s_finite_all.max().item()
                max_pos = torch.where(scores_raw[b0] == max_val)
                if len(max_pos[0]) > 0:
                    mh, mr, mc = (
                        max_pos[0][0].item(),
                        max_pos[1][0].item(),
                        max_pos[2][0].item(),
                    )
                    print(
                        f"\n  Max scores_raw over all heads: {max_val:.2f} at head={mh}, "
                        f"row={mr}, col={mc}"
                    )
                    q_at_max = Q[b0, mh, mr, :]
                    k_at_max = K[b0, mh, mc, :]
                    q_top2, qi2 = q_at_max.abs().topk(2)
                    k_top2, ki2 = k_at_max.abs().topk(2)
                    print(
                        f"    Q max-dims: dim={qi2[0].item()}(val={q_at_max[qi2[0]].item():.2f}), "
                        f"dim={qi2[1].item()}(val={q_at_max[qi2[1]].item():.2f})"
                    )
                    print(
                        f"    K max-dims: dim={ki2[0].item()}(val={k_at_max[ki2[0]].item():.2f}), "
                        f"dim={ki2[1].item()}(val={k_at_max[ki2[1]].item():.2f})"
                    )

            # ── Score scale analysis ──
            raw_val = (
                scores_raw[b0, h0, r0, c0].item()
                if not (
                    torch.isnan(scores_raw[b0, h0, r0, c0])
                    or torch.isinf(scores_raw[b0, h0, r0, c0])
                )
                else float("nan")
            )
            print(
                f"\n  Score at detected position:  raw QK^T={raw_val:.2f}  "
                f"scaled={raw_val / math.sqrt(Q.shape[-1]) if not math.isnan(raw_val) else float('nan'):.2f}"
            )
            print(
                f"  Float32 max={torch.finfo(torch.float32).max:.2e}, "
                f"exp overflow at score > {math.log(torch.finfo(torch.float32).max):.1f}"
            )
            print(f"  {'─' * 60}\n")
            return


_measure = Measure()
_measure._nan_dumped = False


def _install_patches():
    """Monkey-patch encoder layer forward + register FFN/Conv hooks."""
    original_layer_forward = Wav2Vec2BertEncoderLayer.forward

    def patched_layer_forward(
        self,
        hidden_states,
        attention_mask=None,
        relative_position_embeddings=None,
        output_attentions=False,
        conv_attention_mask=None,
    ):
        layer_idx = _measure._current_layer + 1
        _measure.set_layer(layer_idx)

        # ── input ──
        _measure.record(hidden_states, "input")

        # ── FFN1 sub-block ──
        residual = hidden_states
        _measure.record(hidden_states, "ffn1_ln_in")
        hidden_states = self.ffn1_layer_norm(hidden_states)
        _measure.record(hidden_states, "ffn1_ln_out")
        hidden_states = self.ffn1(hidden_states)
        _measure.record(hidden_states, "ffn1_out")
        hidden_states = hidden_states * 0.5 + residual
        _measure.record(hidden_states, "ffn1_res")

        # ── Self-Attention sub-block ──
        residual = hidden_states
        _measure.record(hidden_states, "attn_ln_in")
        hidden_states = self.self_attn_layer_norm(hidden_states)
        _measure.record(hidden_states, "attn_ln_out")

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            relative_position_embeddings=relative_position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        _measure.record(hidden_states, "attn_out")
        hidden_states = hidden_states + residual
        _measure.record(hidden_states, "attn_res")

        # ── Convolution sub-block ──
        residual = hidden_states
        _measure.record(hidden_states, "conv_ln_in")
        hidden_states = self.conv_module(
            hidden_states, attention_mask=conv_attention_mask
        )
        _measure.record(hidden_states, "conv_out")
        hidden_states = residual + hidden_states
        _measure.record(hidden_states, "conv_res")

        # ── FFN2 sub-block ──
        residual = hidden_states
        _measure.record(hidden_states, "ffn2_ln_in")
        hidden_states = self.ffn2_layer_norm(hidden_states)
        _measure.record(hidden_states, "ffn2_ln_out")
        hidden_states = self.ffn2(hidden_states)
        _measure.record(hidden_states, "ffn2_out")
        hidden_states = hidden_states * 0.5 + residual
        _measure.record(hidden_states, "ffn2_res")

        # ── Final LayerNorm ──
        hidden_states = self.final_layer_norm(hidden_states)
        _measure.record(hidden_states, "final_ln")

        return hidden_states, attn_weights

    Wav2Vec2BertEncoderLayer.forward = patched_layer_forward

    # ── FFN hooks for internal (intermediate dense & activation) ──
    def _ffn_hook_pre_act(module, input, output):
        layer_idx = _measure._current_layer
        # Identify which FFN this is (ffn1 vs ffn2) using module reference
        if hasattr(module, "_ffn_name"):
            _measure.record(input[0], f"{module._ffn_name}_pre_act")
            _measure.record(output, f"{module._ffn_name}_post_inter")

    def _ffn_hook_output(module, input, output):
        layer_idx = _measure._current_layer
        if hasattr(module, "_ffn_name"):
            pass  # already recorded at ffnX_out level

    # Register hooks on all existing and future FFN modules
    _original_ffn_init = Wav2Vec2BertFeedForward.__init__

    def _patched_ffn_init(self, config, act_fn=None, hidden_size=None):
        _original_ffn_init(self, config, act_fn, hidden_size)
        self.intermediate_dense.register_forward_hook(_ffn_hook_pre_act)

    Wav2Vec2BertFeedForward.__init__ = _patched_ffn_init

    # ── Conv module hooks ──
    # We need to hook the conv module inside the conv_module
    from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import (
        Wav2Vec2BertConvolutionModule,
        Wav2Vec2BertSelfAttention,
    )

    def _conv_hook_ln(module, input, output):
        layer_idx = _measure._current_layer
        _measure.record(output, "conv_ln_out")

    def _conv_hook_pre_glu(module, input, output):
        layer_idx = _measure._current_layer
        _measure.record(output, "conv_pre_glu")

    def _conv_hook_post_glu(module, input, output):
        layer_idx = _measure._current_layer
        _measure.record(output, "conv_post_glu")

    def _conv_hook_post_depth(module, input, output):
        layer_idx = _measure._current_layer
        _measure.record(output, "conv_post_depth")

    def _conv_hook_post_act(module, input, output):
        layer_idx = _measure._current_layer
        _measure.record(output, "conv_post_act")

    # Monkey-patch conv module forward instead of hooks for cleaner capture
    _original_conv_forward = Wav2Vec2BertConvolutionModule.forward

    def _patched_conv_forward(self, hidden_states, attention_mask=None):
        hidden_states = self.layer_norm(hidden_states)
        _measure.record(hidden_states, "conv_ln_out")

        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(
                ~attention_mask.bool().unsqueeze(-1), 0.0
            )

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.pointwise_conv1(hidden_states)
        _measure.record(hidden_states, "conv_pre_glu")
        hidden_states = self.glu(hidden_states)
        _measure.record(hidden_states, "conv_post_glu")

        hidden_states = torch.nn.functional.pad(
            hidden_states, (self.depthwise_conv.kernel_size[0] - 1, 0)
        )
        hidden_states = self.depthwise_conv(hidden_states)
        _measure.record(hidden_states, "conv_post_depth")

        hidden_states = self.depthwise_layer_norm(
            hidden_states.transpose(1, 2)
        ).transpose(1, 2)
        hidden_states = self.activation(hidden_states)
        _measure.record(hidden_states, "conv_post_act")

        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        _measure.record(hidden_states, "conv_out")
        return hidden_states

    Wav2Vec2BertConvolutionModule.forward = _patched_conv_forward

    # ── Attention internal instrumentation ──
    import math

    _original_attn_forward = Wav2Vec2BertSelfAttention.forward

    def _patched_attn_forward(
        self,
        hidden_states,
        attention_mask=None,
        relative_position_embeddings=None,
        output_attentions=False,
    ):
        measure = _measure
        layer = measure._current_layer

        batch_size, seq_len, hsize = hidden_states.shape
        query_key_states = hidden_states
        value_states = hidden_states

        if self.position_embeddings_type == "rotary":
            if relative_position_embeddings is None:
                raise ValueError(
                    "`relative_position_embeddings` has to be defined when "
                    "`self.position_embeddings_type == 'rotary'"
                )
            query_key_states = self._apply_rotary_embedding(
                query_key_states, relative_position_embeddings
            )

        # ── Q, K, V projections ──
        Q_proj = self.linear_q(query_key_states)  # (batch, seq, hidden)
        K_proj = self.linear_k(query_key_states)
        V_proj = self.linear_v(value_states)
        measure.record(Q_proj, "attn_Q_proj")
        measure.record(K_proj, "attn_K_proj")
        measure.record(V_proj, "attn_V_proj")

        # Reshape to (batch, heads, seq, head_dim)
        Q = Q_proj.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        K = K_proj.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        V = V_proj.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        measure.record(Q, "attn_Q_head")
        measure.record(K, "attn_K_head")

        # Per-head max
        q_per_head = Q.abs().amax(dim=(-1, -2))
        k_per_head = K.abs().amax(dim=(-1, -2))
        measure.record(q_per_head, "attn_Q_per_head")

        # ── Raw scores ──
        scores_raw = torch.matmul(Q, K.transpose(-2, -1))  # (batch, heads, seq, seq)
        measure.record(scores_raw, "attn_scores_raw")

        # ── Scaled scores ──
        scores = scores_raw / math.sqrt(self.head_size)
        measure.record(scores, "attn_scores")

        # ── Relative position (relative_key) ──
        scores_dist = None
        if self.position_embeddings_type == "relative":
            if relative_position_embeddings is None:
                raise ValueError(
                    "`relative_position_embeddings` has to be defined when "
                    "`self.position_embeddings_type == 'relative'"
                )
            scores = self._apply_relative_embeddings(
                query=Q,
                key=K,
                relative_position_embeddings=relative_position_embeddings,
            )
            measure.record(scores, "attn_scores_rel")

        if self.position_embeddings_type == "relative_key":
            q_len, k_len = Q.shape[2], K.shape[2]
            pos_ids_l = torch.arange(q_len, device=hidden_states.device).view(-1, 1)
            pos_ids_r = torch.arange(k_len, device=hidden_states.device).view(1, -1)
            distance = pos_ids_r - pos_ids_l
            distance = torch.clamp(
                distance,
                -self.left_max_position_embeddings,
                self.right_max_position_embeddings,
            )
            pos_emb = self.distance_embedding(
                distance + self.left_max_position_embeddings
            )
            pos_emb = pos_emb.to(dtype=Q.dtype)
            rel_weights = torch.einsum("bhld,lrd->bhlr", Q, pos_emb)
            measure.record(rel_weights, "attn_dist_raw")
            scores = scores + (rel_weights / math.sqrt(self.head_size))
            scores_dist = scores
            measure.record(scores, "attn_scores_dist")

        # ── Attention mask ──
        if attention_mask is not None:
            scores = scores + attention_mask
            measure.record(scores, "attn_scores_masked")

        # ── Softmax ──
        probs = torch.softmax(scores, dim=-1)
        measure.record(probs, "attn_probs")

        probs = self.dropout(probs)

        # ── Context ──
        context = torch.matmul(probs, V)
        measure.record(context, "attn_context")

        # ── Output projection ──
        context = context.transpose(1, 2).reshape(
            batch_size, -1, self.num_heads * self.head_size
        )
        output = self.linear_out(context)
        measure.record(output, "attn_out_internal")

        # ── NaN dump (first occurrence only) ──
        _dump_attn_nan(measure, layer, Q, K, scores_raw, scores, probs, scores_dist)

        return output, probs

    Wav2Vec2BertSelfAttention.forward = _patched_attn_forward

    # Tag ffn modules so hooks know which is which
    _original_ffn_name_init = Wav2Vec2BertFeedForward.__init__

    # Actually let me handle FFN naming differently. We'll tag them after construction.
    # Better: use closure over layer_idx.
    # Let me use a post-init hook on the encoder layer.

    # ── Tag FFNs after encoder layer init ──
    _original_layer_init = Wav2Vec2BertEncoderLayer.__init__

    def _patched_layer_init(self, config):
        _original_layer_init(self, config)
        # Tag the ffn modules so hooks can identify them
        self.ffn1._ffn_name = "ffn1"
        self.ffn2._ffn_name = "ffn2"
        # Register hooks on intermediate dense layers
        self.ffn1.intermediate_dense._ffn_name = "ffn1"
        self.ffn2.intermediate_dense._ffn_name = "ffn2"

    Wav2Vec2BertEncoderLayer.__init__ = _patched_layer_init


# ── build model ───────────────────────────────────────────────────────────────


def _reinit_distance_embeddings(model, init_std):
    """Safety net: re-init distance_embedding with small std to prevent NaN."""
    for module in model.modules():
        if hasattr(module, "distance_embedding") and isinstance(module.distance_embedding, nn.Embedding):
            nn.init.normal_(module.distance_embedding.weight, mean=0.0, std=init_std)


def build_model(init_range, num_layers):
    config = Wav2Vec2BertForRNNStreamingMultilevelCTCConfig(
        level_to_vocab_size=_level_to_vocab_size,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=num_layers,
        rnn_hidden_size=128,
        add_adapter=False,
        apply_spec_augment=False,
        mask_time_prob=0.0,
        layerdrop=0.0,
        conv_depthwise_kernel_size=9,
        initializer_range=init_range,
        layer_norm_eps=1e-3,
        ctc_zero_infinity=True,
    )
    model = Wav2Vec2BertForRNNStreamingMultilevelCTC.from_pretrained(
        "facebook/w2v-bert-2.0",
        config=config,
        ignore_mismatched_sizes=True,
    )
    _reinit_distance_embeddings(model, config.initializer_range)
    model.eval()
    return model


# ── run and print ─────────────────────────────────────────────────────────────


def run_test(init_range, num_layers, audio_type, seed=0, label=""):
    global _measure
    _measure = Measure()
    _measure._nan_dumped = False

    model = build_model(init_range, num_layers)
    processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    if audio_type == "silence":
        inputs = processor(
            [[0.0] * 16000] * 2, return_tensors="pt", sampling_rate=16000
        )
    else:
        torch.manual_seed(seed + 1000)
        audio = [
            (
                torch.sin(torch.linspace(0, 200 * np.pi, 16000)) * 0.3
                + torch.randn(16000) * 0.05
            )
            .clamp(-1, 1)
            .numpy()
            for _ in range(2)
        ]
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

    print(f"\n{'=' * 100}")
    print(f"  {label}")
    print(f"{'=' * 100}\n")

    with torch.no_grad():
        out = model.wav2vec2_bert(
            inputs["input_features"],
            attention_mask=inputs.get("attention_mask"),
        )

    any_nan = torch.isnan(out.last_hidden_state).any().item()
    any_inf = torch.isinf(out.last_hidden_state).any().item()
    final_max = out.last_hidden_state.abs().max().item()

    # Print per-step table
    for row in _measure.table_rows():
        print(row)

    # Print amplification factors
    print(f"\n  ── Amplification factors (σ_out / σ_in) per submodule ──")
    for row in _measure.amplification_table():
        print(row)

    # Summary
    print(
        f"\n  Final output: max_abs={final_max:.4f}, has_inf={any_inf}, has_nan={any_nan}"
    )

    # ── First NaN trace ──
    nan_records = [(k, v) for k, v in _measure.records.items() if v["has_nan"]]
    if nan_records:
        nan_records.sort(
            key=lambda x: (x[0][0], list(_measure.records.keys()).index(x[0]))
        )
        print(
            f"\n  ◄◆ FIRST NaN detected at: layer={nan_records[0][0][0]}, step='{nan_records[0][0][1]}'"
        )
        print(f"     Tensor stats: {nan_records[0][1]}")
        # Also check the previous step
        prev = None
        for k, v in _measure.records.items():
            if k == nan_records[0][0]:
                break
            prev = v
        if prev:
            print(
                f"     Previous step ({prev['label']}):  max_abs={prev['max_abs']:.4f}, std={prev['std']:.4f}, "
                f"max={prev['max']:.4f}, min={prev['min']:.4f}"
            )
        # Show all early warnings (steps with max_abs > 100)
        print(f"\n  ── Precursors (max_abs > 10) ──")
        for k, v in _measure.records.items():
            if v["max_abs"] > 10 and not v["has_nan"]:
                print(
                    f"     layer={k[0]:>2}, step={k[1]:<18}  max_abs={v['max_abs']:.2f}  std={v['std']:.2f}"
                )
            if v["has_inf"] and not v["has_nan"]:
                print(
                    f"     layer={k[0]:>2}, step={k[1]:<18}  ◄── Inf appears here first!"
                )

    return _measure


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _install_patches()

    # Compare 16-layer vs 8-layer on silence (worst case)
    run_test(
        init_range=0.02,
        num_layers=16,
        audio_type="silence",
        seed=100,
        label="init=0.02, 16 layers, silence [seed=0]",
    )

    # run_test(
    #     init_range=0.02,
    #     num_layers=16,
    #     audio_type="audio",
    #     seed=0,
    #     label="init=0.02, 16 layers, audio [seed=0]",
    # )
    #
    # run_test(
    #     init_range=0.02,
    #     num_layers=8,
    #     audio_type="silence",
    #     seed=0,
    #     label="init=0.02, 8 layers, silence [seed=0]",
    # )

    # # Run 16-layer with a few seeds to show variability
    # for s in [1, 2, 5]:
    #     run_test(init_range=0.02, num_layers=16, audio_type="audio",
    #              seed=s, label=f"init=0.02, 16 layers, audio [seed={s}]")
