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
from typing import TYPE_CHECKING, Optional

from sglang.srt.environ import envs
from sglang.srt.utils import DynamicGradMode

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler


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
        # ``pending_*`` tuple layout:
        #   (lazy_tokens, prefill, decode, mode, batch_copy, reqs)
        pending_curr: Optional[tuple] = None
        pending_next: Optional[tuple] = None

        def _finalize(pending: tuple, extract_cache: bool):
            _, pref, dec, mode, batch_copy, reqs_snapshot = pending
            result = self.tp_worker.finalize_mlx_result(
                pref,
                dec,
                mode,
                reqs_snapshot,
                extract_cache=extract_cache,
            )
            if result.next_token_ids is not None:
                batch_copy.output_ids = result.next_token_ids
            self.process_batch_result(batch_copy, result)

        def _launch_fresh(batch: ScheduleBatch) -> tuple:
            mwb = batch.get_model_worker_batch()
            lazy_tokens, pref, dec, mode = (
                self.tp_worker.async_forward_batch_generation_mlx(mwb)
            )
            return (
                lazy_tokens,
                pref,
                dec,
                mode,
                batch.copy(),
                list(batch.reqs),
            )

        def _launch_chained(prev: tuple) -> tuple:
            prev_decode = prev[2]  # MlxPendingDecode
            prev_batch_copy = prev[4]
            prev_reqs = prev[5]
            lazy_tokens, pref, dec, mode = self.tp_worker.async_chained_decode_mlx(
                prev_decode
            )
            # Composition is identical to prev: reuse a fresh batch copy
            # of the same underlying ScheduleBatch so process_batch_result
            # updates the same req objects with the new token.
            return (
                lazy_tokens,
                pref,
                dec,
                mode,
                prev_batch_copy.copy(),
                prev_reqs,
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
                and pending_curr[3] == "decode"
                and not self.waiting_queue
            )
            if can_chain and pending_next is None:
                # Build + launch the chained step BEFORE we block on
                # pending_curr — this is the "no idle gap" trick.
                pending_next = _launch_chained(
                    pending_curr
                )  # GPU now has 2 steps queued
                self.result_queue.append(pending_next)

            # 2. Finalize/process on pending_curr's tokens. Because pending_next is still
            #    referencing curr's batch_cache, extract_cache=False.
            if pending_curr is not None:
                _finalize(pending_curr, extract_cache=(pending_next is None))
                self.result_queue.popleft()
                pending_curr = None
            # ^ while this block returns, GPU is executing pending_next.

            # 3. Decide whether pending_next is still valid and promote it.
            finished_any = any(
                req.finished() for req in (pending_next[5] if pending_next else [])
            )
            new_prefill_waiting = bool(self.waiting_queue)
            if (
                pending_next is not None
                and not finished_any
                and not new_prefill_waiting
            ):
                pending_curr = pending_next
                pending_next = None
                self.cur_batch = pending_curr[4]
                self.last_batch = pending_curr[4]
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
