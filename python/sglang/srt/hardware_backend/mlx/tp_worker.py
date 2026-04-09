"""MLX-specific TpModelWorker subclass for Apple Silicon.

Overrides the standard TpModelWorker to route forward passes through
the native MLX model runner, avoiding PyTorch MPS entirely for inference.

PyTorch model weights are never loaded.  A lightweight ModelRunner stub
(MlxModelRunnerStub) provides only the minimal bookkeeping structures
(req_to_token_pool, token_to_kv_pool_allocator with a zero-memory
dummy KV cache) that the SGLang scheduler expects.  The actual KV cache
is managed internally by the MLX model runner.
"""

import logging
from typing import Optional, Union

import mlx.core as mx
import torch

from sglang.srt.hardware_backend.mlx.model_runner import (
    MlxPendingDecode,
    MlxPendingPrefill,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

logger = logging.getLogger(__name__)


class MlxTpModelWorker(TpModelWorker):
    """A tensor parallel model worker that routes inference through MLX.

    Inherits from TpModelWorker for scheduler integration, but replaces
    the standard ModelRunner with MlxModelRunnerStub (no PyTorch weights,
    zero-memory KV cache) and delegates all forward passes to a native
    MlxModelRunner.
    """

    def _init_model_runner(self):
        """Override to use a lightweight ModelRunner that skips weight loading."""
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
        from sglang.srt.hardware_backend.mlx.model_runner_stub import (
            MlxModelRunnerStub,
        )

        self._model_runner = MlxModelRunnerStub(
            model_config=self.model_config,
            mem_fraction_static=self.server_args.mem_fraction_static,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            moe_ep_rank=self.moe_ep_rank,
            moe_ep_size=self.ep_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            nccl_port=self.nccl_port,
            dp_rank=self.dp_rank,
            server_args=self.server_args,
            is_draft_worker=self.is_draft_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            memory_pool_config=self.memory_pool_config,
        )

        # Initialize the MLX model runner (loads weights via MLX, not PyTorch)
        logger.info("Initializing MlxModelRunner for end-to-end MLX inference")
        self._mlx_runner = MlxModelRunner(
            model_path=self.server_args.model_path,
            trust_remote_code=self.server_args.trust_remote_code,
        )
        self._mlx_active_rids: set[str] = set()

    def get_pad_input_ids_func(self):
        """Override since the stub ModelRunner has no real model."""
        return None

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        forward_batch: Optional[ForwardBatch] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        is_verify: bool = False,
        skip_attn_backend_init=False,
    ) -> GenerationBatchResult:
        """Override to route through MLX model runner."""
        if model_worker_batch is not None:
            return self._forward_batch_generation_mlx(model_worker_batch)

        # Fallback to standard path for None batches
        return super().forward_batch_generation(
            model_worker_batch,
            forward_batch,
            pp_proxy_tensors,
            is_verify,
            skip_attn_backend_init,
        )

    def _forward_batch_generation_mlx(
        self,
        model_worker_batch: ModelWorkerBatch,
    ) -> GenerationBatchResult:
        """Run forward pass through the MLX model runner.

        Bypasses the standard ModelRunner forward+sample and uses native MLX
        inference for the entire model. Only supports greedy sampling.
        """
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        forward_mode = model_worker_batch.forward_mode
        reqs = model_worker_batch.reqs

        if forward_mode.is_idle():
            return GenerationBatchResult(
                logits_output=LogitsProcessorOutput(next_token_logits=None),
                can_run_cuda_graph=False,
            )

        # Auto-cleanup: remove MLX state for requests no longer in the batch
        current_rids = {req.rid for req in reqs}
        stale_rids = self._mlx_active_rids - current_rids
        for rid in stale_rids:
            self._mlx_runner.remove_request(rid)
        self._mlx_active_rids = current_rids

        next_token_ids_list = []

        if forward_mode.is_extend():
            # Prefill (or MIXED): extract per-request tokens from concatenated input_ids
            input_ids_cpu = model_worker_batch.input_ids.cpu().tolist()
            extend_seq_lens = model_worker_batch.extend_seq_lens
            offset = 0
            prefill_rids = []
            decode_rids = []
            for i, req in enumerate(reqs):
                seq_len = extend_seq_lens[i]
                req_token_ids = input_ids_cpu[offset : offset + seq_len]
                offset += seq_len
                if req.rid in self._mlx_runner._request_states:
                    # MIXED mode: this request already has MLX state, decode it
                    decode_rids.append(req.rid)
                else:
                    # Prefill: new request
                    next_token = self._mlx_runner.prefill(req.rid, req_token_ids)
                    prefill_rids.append((req.rid, next_token))

            # Batch decode all existing requests at once
            if decode_rids:
                decode_results = self._mlx_runner.decode_batch(decode_rids)
                decode_map = dict(zip(decode_rids, decode_results))
            else:
                decode_map = {}

            prefill_map = dict(prefill_rids)

            # Reassemble in original request order
            for req in reqs:
                if req.rid in decode_map:
                    next_token_ids_list.append(decode_map[req.rid])
                else:
                    next_token_ids_list.append(prefill_map[req.rid])

        elif forward_mode.is_decode():
            # Decode: batch decode all requests
            req_ids = [req.rid for req in reqs]
            next_token_ids_list = self._mlx_runner.decode_batch(req_ids)

        else:
            raise ValueError(
                f"MLX runner does not support forward mode: {forward_mode}"
            )

        next_token_ids = torch.tensor(
            next_token_ids_list, dtype=torch.long, device="cpu"
        )

        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=next_token_ids,
            can_run_cuda_graph=False,
        )

    def async_forward_batch_generation_mlx(
        self,
        model_worker_batch: ModelWorkerBatch,
    ) -> tuple[
        Union[mx.array, None],
        Optional[list[MlxPendingPrefill]],
        Optional[MlxPendingDecode],
        str,
    ]:
        """Start an async (lazy) forward pass through the MLX model runner.

        Returns ``(lazy_tokens, pending_state, decode_reqs, mode)`` where
        ``lazy_tokens`` is an unevaluated ``mx.array``.  The caller should
        call ``mx.async_eval(lazy_tokens)`` to kick off GPU work, do CPU
        scheduling, then call :meth:`finalize_mlx_result` to materialise.

        For idle batches all four values are ``None / "idle"``.
        """
        forward_mode = model_worker_batch.forward_mode
        reqs = model_worker_batch.reqs

        if forward_mode.is_idle():
            return None, None, None, "idle"

        # Auto-cleanup stale requests
        current_rids = {req.rid for req in reqs}
        stale_rids = self._mlx_active_rids - current_rids
        for rid in stale_rids:
            self._mlx_runner.remove_request(rid)
        self._mlx_active_rids = current_rids

        if forward_mode.is_decode():
            req_ids = [req.rid for req in reqs]
            pending_decode = self._mlx_runner.decode_batch_start(req_ids)
            mx.async_eval(pending_decode.lazy_tokens, *pending_decode.batch_cache)
            return pending_decode.lazy_tokens, None, pending_decode, "decode"

        elif forward_mode.is_extend():
            # Prefill (EXTEND or MIXED): start each new request lazily and
            # collect the lazy tokens + KV cache states for async eval.

            # We must include cache states in mx.async_eval, not just the
            # logit token. The KV cache is updated as a side effect of the
            # model call; mx.async_eval on only the token does NOT guarantee
            # cache state is materialised before the next decode step reads it.
            input_ids_cpu = model_worker_batch.input_ids.cpu().tolist()
            extend_seq_lens = model_worker_batch.extend_seq_lens
            offset = 0
            pending_prefills: list[MlxPendingPrefill] = []
            mixed_decode_rids: list[str] = []

            # Process requests
            for i, req in enumerate(reqs):
                seq_len = extend_seq_lens[i]
                req_token_ids = input_ids_cpu[offset : offset + seq_len]
                offset += seq_len
                if req.rid in self._mlx_runner._request_states:
                    # MIXED mode: already has KV state -> decode lazily.
                    mixed_decode_rids.append(req.rid)
                else:
                    # Add prefill forward pass without evaluating
                    pending_prefills.append(
                        self._mlx_runner.prefill_start(req.rid, req_token_ids)
                    )

            # Start mixed-mode decode lazily so it can overlap with CPU scheduling
            pending_mixed_decode: Optional[MlxPendingDecode] = None
            if mixed_decode_rids:
                pending_mixed_decode = self._mlx_runner.decode_batch_start(
                    mixed_decode_rids
                )

            # Kick off GPU work
            # Prefill requests present
            if pending_prefills:
                # Stack prefill tokens
                lazy_stacked = mx.stack(
                    [p.lazy_token for p in pending_prefills], axis=0
                )
                # Collect every cache layer state so the GPU also materialises
                # the KV cache before the next decode step starts.
                cache_states = [c.state for p in pending_prefills for c in p.cache]
                # Decode requests present
                if pending_mixed_decode is not None:
                    mx.async_eval(
                        lazy_stacked,
                        pending_mixed_decode.lazy_tokens,
                        *cache_states,
                        *pending_mixed_decode.batch_cache,
                    )
                # All requests are prefills
                else:
                    mx.async_eval(lazy_stacked, *cache_states)
            # All requests are mixed-mode decodes; no prefills
            elif pending_mixed_decode is not None:
                mx.async_eval(
                    pending_mixed_decode.lazy_tokens, *pending_mixed_decode.batch_cache
                )
                lazy_stacked = None
            # No requests present
            else:
                lazy_stacked = None

            return lazy_stacked, pending_prefills, pending_mixed_decode, "extend"

        raise ValueError(
            f"MLX async runner does not support forward mode: {forward_mode}"
        )

    def async_chained_decode_mlx(
        self,
        prev_pending: MlxPendingDecode,
    ) -> tuple[mx.array, None, MlxPendingDecode, str]:
        """Launch a decode step that chains off a still-lazy previous decode.

        This is the "no idle gap" pipelining primitive: we build the next
        decode's compute graph using ``prev_pending.lazy_tokens`` (still
        unevaluated) as its input ids, kick the combined graph off with
        ``mx.async_eval``, and return. The GPU runs the new step immediately
        after ``prev_pending`` with no scheduling gap, while the caller is
        free to block on ``prev_pending`` and run CPU-side bookkeeping.

        Preconditions (caller must ensure):
        * ``prev_pending`` was produced by a previous decode start (either
          :meth:`async_forward_batch_generation_mlx` in decode mode or a
          previous :meth:`async_chained_decode_mlx`) and has NOT yet been
          finalised with ``extract_cache=True``.
        * The batch composition for this step is identical to
          ``prev_pending`` — same requests, same order. Composition changes
          (finished reqs, new prefills) must break the chain instead.

        Returns a 4-tuple matching the shape of
        :meth:`async_forward_batch_generation_mlx` for the decode case:
        ``(lazy_tokens, None, pending_decode, "decode")``. The ``None`` slot
        is the prefill list, which is always absent for chained decodes.
        """
        # Pass in previous decode request and build computation graph for next decode
        pending = self._mlx_runner.decode_batch_start_chained(prev_pending)
        mx.async_eval(pending.lazy_tokens, *pending.batch_cache)
        return pending.lazy_tokens, None, pending, "decode"

    def finalize_mlx_result(
        self,
        prefill: Optional[list[MlxPendingPrefill]],
        decode: Optional[MlxPendingDecode],
        mode: str,
        reqs: list,
        extract_cache: bool = True,
    ) -> GenerationBatchResult:
        """Materialise a lazy MLX result into a :class:`GenerationBatchResult`.

        The blocking wait happens inside ``decode_batch_finalize`` /
        ``prefill_finalize`` via ``.tolist()`` / ``.item()`` on the specific
        lazy outputs.

        Args:
            prefill: List of ``MlxPendingPrefill`` to finalise (extend mode).
            decode: The ``MlxPendingDecode`` to finalise (decode or mixed
                mode).
            mode: One of ``"decode"``, ``"extend"``, ``"idle"``.
            reqs: The request list in the original batch order, used to
                reassemble per-request next tokens for mixed extend batches.
            extract_cache: Forwarded to :meth:`MlxModelRunner.decode_batch_finalize`.
                Pass ``False`` when this pending is mid-chain (another
                chained decode has already been launched on top of its
                shared ``batch_cache``); pass ``True`` for the tail of a
                chain so per-request caches are snapshotted before any
                non-chained op runs on them.
        """
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        if mode == "idle":
            return GenerationBatchResult(
                logits_output=LogitsProcessorOutput(next_token_logits=None),
                can_run_cuda_graph=False,
            )

        if mode == "decode":
            next_tokens_list = self._mlx_runner.decode_batch_finalize(
                decode, extract_cache=extract_cache
            )

        elif mode == "extend":
            # Finalise each prefill
            if prefill:
                prefill_tokens: list[int] = []
                for pending_p in prefill:
                    tok = self._mlx_runner.prefill_finalize(pending_p)
                    prefill_tokens.append(tok)
                prefill_map = {p.req_id: t for p, t in zip(prefill, prefill_tokens)}
            else:
                prefill_map = {}

            # Finalize mixed-mode decodes and update per-request state
            if decode:
                decode_results: dict[str, int] = {}
                if decode is not None:
                    mixed_tokens = self._mlx_runner.decode_batch_finalize(
                        decode, extract_cache=extract_cache
                    )
                    decode_results = {
                        rid: tok
                        for (rid, _), tok in zip(decode.decode_reqs, mixed_tokens)
                    }
            else:
                decode_results = {}

            next_tokens_list = []
            for req in reqs:
                if req.rid in prefill_map:
                    next_tokens_list.append(prefill_map[req.rid])
                else:
                    next_tokens_list.append(decode_results[req.rid])

        else:
            raise ValueError(f"Unknown MLX async mode: {mode}")

        next_token_ids = torch.tensor(next_tokens_list, dtype=torch.long, device="cpu")
        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=next_token_ids,
            can_run_cuda_graph=False,
        )
