"""MLX overlap scheduling mixin for the SGLang scheduler.

Provides ``event_loop_overlap_mlx`` and ``_run_batch_mlx_overlap`` which
are structurally identical to the normal event loop but force in-place
tensor operations in ``prepare_for_decode``.

Non-in-place MPS tensor allocations (e.g. ``seq_lens + 1`` instead of
``seq_lens.add_(1)``) create fresh Metal buffers each decode step.
These interfere with the MLX Metal command stream, adding ~8 ms per
decode step.  Forcing in-place ops eliminates this overhead and brings
overlap ON performance to parity with overlap OFF.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import mlx.core as mx

from sglang.srt.environ import envs
from sglang.srt.utils import DynamicGradMode

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.mlx.model_runner import (
        MlxPendingDecode,
        MlxPendingPrefill,
    )
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler


@dataclass
class MlxPendingJob:
    """Unfinished MLX work and graphs on the generation stream.

    Attributes:
        lazy_tokens: Lazily evaluated token IDs produced by the forward pass.
            This array has not been materialised yet; calling ``.tolist()`` or
            ``mx.eval()`` on it will block until the Metal kernel completes.
        prefill: MLX prefill state returned by the model worker.  Contains
            per-layer KV-write descriptors needed by
            ``finalize_mlx_result`` to commit prefill caches.  Empty list for
            pure-decode steps.
        decode: Opaque decode state used to chain the *next* decode step via
            ``async_chained_decode_mlx`` without blocking on ``lazy_tokens``
            first.  Encapsulates the KV-cache handles and sampling state so
            the follow-on graph can be built while the current one is still
            running on the GPU.
        mode: Either ``"prefill"`` or ``"decode"``, describing which forward
            pass produced this job.  Used by ``finalize_mlx_result`` to choose
            the right result-extraction path and by the overlap loop to decide
            whether chaining is safe.
        batch_copy: Snapshot of the :class:`ScheduleBatch` at the moment this
            job was launched.  Decoupled from the live batch so that
            ``process_batch_result`` can update request state without racing
            against the scheduler's next scheduling decision.
        reqs: Snapshot of ``batch.reqs`` at launch time.  Kept separately so
            that the overlap loop can check ``req.finished()`` on the *previous*
            step's request list without holding a reference to the mutable
            batch object.
    """

    lazy_tokens: mx.array
    prefill: list["MlxPendingPrefill"]
    decode: "MlxPendingDecode"
    mode: str
    batch_copy: "ScheduleBatch"
    reqs: List[Req]


class SchedulerMlxOverlapMixin:
    """Mixin that adds MLX overlap scheduling to :class:`Scheduler`."""

    @DynamicGradMode()
    def event_loop_overlap_mlx(self: "Scheduler"):
        """MLX-specific overlap loop modelled on ``mlx_lm.generate.generate_step``.

        At steady state we keep TWO in-flight MLX graphs on the generation
        stream:

        * ``pending_curr`` — the step whose tokens we are about to block on
          and feed into the scheduler's bookkeeping.
        * ``pending_next`` — the step that was built on top of
          ``pending_curr``'s still-lazy output tokens via
          ``async_chained_decode_mlx`` and has already been handed to
          ``mx.async_eval``. Because MLX tracks the full dependency graph,
          the GPU will execute ``pending_next`` back-to-back with
          ``pending_curr`` — there is no scheduling gap on the device.

        Bookkeeping timeline for a steady-state decode loop:

            iter k:
              build pending_next  (CPU graph build + mx.async_eval; cheap)
              block on pending_curr via .tolist() (wait only on curr's tokens)
              process_batch_result(pending_curr)   <-- GPU is running pending_next
              pending_curr = pending_next

        The chain is broken (we fall back to a "schedule + launch" step)
        whenever any of the following holds:

        * ``pending_curr`` is not a pure decode (e.g. prefill/extend).
        * The waiting queue has new requests that need prefill.
        * Any req in ``pending_curr`` just finished this iteration, so the
          composition for ``pending_next`` would need to shrink.

        When the chain breaks mid-flight we still finalise the already-
        launched ``pending_next`` normally (its tokens are valid for all
        surviving reqs). We pass ``extract_cache=True`` to the LAST finalise
        in a chain so per-request caches get snapshotted back into
        ``state.cache`` before any non-chained op runs on them.
        """
        pending_curr: Optional[MlxPendingJob] = None
        pending_next: Optional[MlxPendingJob] = None

        def _finalize(pending: MlxPendingJob, extract_cache: bool):
            result = self.tp_worker.finalize_mlx_result(
                pending.prefill,
                pending.decode,
                pending.mode,
                pending.reqs,
                extract_cache=extract_cache,
            )
            if result.next_token_ids is not None:
                pending.batch_copy.output_ids = result.next_token_ids
            self.process_batch_result(pending.batch_copy, result)

        def _launch_fresh(batch: "ScheduleBatch") -> MlxPendingJob:
            mwb = batch.get_model_worker_batch()
            lazy_tokens, pref, dec, mode = (
                self.tp_worker.async_forward_batch_generation_mlx(mwb)
            )
            return MlxPendingJob(
                lazy_tokens=lazy_tokens,
                prefill=pref,
                decode=dec,
                mode=mode,
                batch_copy=batch.copy(),
                reqs=list(batch.reqs),
            )

        def _launch_chained(prev: MlxPendingJob) -> MlxPendingJob:
            lazy_tokens, pref, dec, mode = self.tp_worker.async_chained_decode_mlx(
                prev.decode
            )
            # Composition is identical to prev: reuse a fresh batch copy
            # of the same underlying ScheduleBatch so process_batch_result
            # updates the same req objects with the new token.
            return MlxPendingJob(
                lazy_tokens=lazy_tokens,
                prefill=pref,
                decode=dec,
                mode=mode,
                batch_copy=prev.batch_copy.copy(),
                reqs=prev.reqs,
            )

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            # 1. If pending_curr is a pure decode AND no new prefill is waiting,
            #    build pending_next on top of it NOW — before we block on curr.
            can_chain = (
                pending_curr is not None
                and pending_curr.mode == "decode"
                and not self.waiting_queue
            )
            if can_chain and pending_next is None:
                # Build + launch the chained step BEFORE we block on
                # pending_curr — this is the "no idle gap" trick.
                # GPU now has 2 steps queued.
                pending_next = _launch_chained(pending_curr)
                self.result_queue.append(pending_next)

            # 2. Finalize/process on pending_curr's tokens. Because pending_next is still
            #    referencing curr's batch_cache, extract_cache=False. (GPU executing pending_next)
            if pending_curr is not None:
                _finalize(pending_curr, extract_cache=(pending_next is None))
                self.result_queue.popleft()
                pending_curr = None

            # 3. Decide whether pending_next is still valid (if no reqs finished)
            #    and promote it.
            finished_any = any(
                req.finished() for req in (pending_next.reqs if pending_next else [])
            )
            new_prefill_waiting = bool(self.waiting_queue)
            if (
                pending_next is not None
                and not finished_any
                and not new_prefill_waiting
            ):
                pending_curr = pending_next
                pending_next = None
                self.cur_batch = pending_curr.batch_copy
                self.last_batch = pending_curr.batch_copy
                if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                    self.self_check_during_busy()
                continue

            # 4. Chain is broken. Finalise pending_next (if any) with extract_cache=True
            #    so per-req caches get snapshotted back, then schedule fresh.
            if pending_next is not None:
                _finalize(pending_next, extract_cache=True)
                self.result_queue.popleft()
                pending_next = None
            next_batch = self.get_next_batch_to_run()
            self.cur_batch = next_batch
            if next_batch:
                pending_curr = _launch_fresh(next_batch)
                self.result_queue.append(pending_curr)
            else:
                self.on_idle()

            self.last_batch = next_batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()
