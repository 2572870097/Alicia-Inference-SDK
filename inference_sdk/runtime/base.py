"""
Base Inference Engine with LeRobot-style Async Inference.

Key Features (LeRobot Pattern):
- Timestamp-aligned action queue (not FIFO)
- Time-based action selection (skip expired actions)
- Queue refill based on chunk threshold
- Observation queue maxsize=1 (always use latest frame)
"""

import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty, Full
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np

from ..core.types import Observation, PolicyMetadata, PolicyStatus
from .monitoring import get_inference_monitor

logger = logging.getLogger(__name__)

@dataclass
class TimedAction:
    """Action with timestamp for time-aligned execution."""
    timestamp: float  # Absolute time when this action should be executed
    timestep: int     # Sequential step index
    action: np.ndarray
    
    def get_timestamp(self) -> float:
        return self.timestamp
    
    def get_timestep(self) -> int:
        return self.timestep
    
    def get_action(self) -> np.ndarray:
        return self.action


@dataclass  
class TimedObservation:
    """Observation with timestamp for control-loop scheduling."""
    timestamp: float  # When observation was captured
    timestep: int     # Sequential step index
    images: Dict[str, np.ndarray]
    state: np.ndarray
    must_go: bool = False  # If True, this observation MUST be processed
    
    def get_timestamp(self) -> float:
        return self.timestamp
    
    def get_timestep(self) -> int:
        return self.timestep


@dataclass
class SmoothingConfig:
    """Configuration for async inference runtime behavior."""
    # Control frequency
    control_fps: float = 30.0

    # Async inference settings
    enable_async_inference: bool = False
    
    # Chunk threshold: trigger new inference when queue_size / chunk_size <= threshold
    chunk_size_threshold: float = 0.5

    # Generic temporal ensembling coefficient. When set, `step()` performs
    # inference on every control step and exponentially blends overlapping
    # future actions, following the ACT temporal ensembling pattern.
    temporal_ensemble_coeff: Optional[float] = None
    
    # Observation queue settings (LeRobot uses maxsize=1)
    obs_queue_maxsize: int = 1
    
    # Fallback when queue empty
    fallback_mode: str = "repeat"  # "repeat", "hold"
    
    @property
    def environment_dt(self) -> float:
        """Time step in seconds."""
        return 1.0 / self.control_fps

    @property
    def temporal_ensemble_enabled(self) -> bool:
        return self.temporal_ensemble_coeff is not None

class TemporalEnsembler:
    """
    Runtime-level temporal ensembling for action chunks.

    This mirrors the ACT temporal ensembling idea, but works on any policy
    that emits a sequence of future actions from `_predict_chunk(...)`.
    """

    def __init__(self, temporal_ensemble_coeff: float):
        if not math.isfinite(temporal_ensemble_coeff):
            raise ValueError("`temporal_ensemble_coeff` must be finite")
        self.temporal_ensemble_coeff = float(temporal_ensemble_coeff)
        self._ensemble_weights: Optional[np.ndarray] = None
        self._ensemble_weights_cumsum: Optional[np.ndarray] = None
        self._weights_capacity: int = 0
        self.reset()

    def reset(self):
        self.ensembled_actions: Optional[np.ndarray] = None
        self.ensembled_actions_count: Optional[np.ndarray] = None

    def _ensure_weights(self, size: int):
        if size <= self._weights_capacity:
            return

        indices = np.arange(size, dtype=np.float32)
        self._ensemble_weights = np.exp(-self.temporal_ensemble_coeff * indices).astype(np.float32)
        self._ensemble_weights_cumsum = np.cumsum(self._ensemble_weights, axis=0)
        self._weights_capacity = size

    def update(self, actions: np.ndarray) -> np.ndarray:
        """
        Update the rolling ensemble and return the next action to execute.

        Args:
            actions: Array with shape (horizon, action_dim) or (action_dim,)
        """
        actions_array = np.asarray(actions, dtype=np.float32)
        if actions_array.ndim == 1:
            actions_array = actions_array[None, :]
        if actions_array.ndim != 2:
            raise ValueError(
                f"`actions` must have shape (horizon, action_dim), got {actions_array.shape}"
            )
        if actions_array.shape[0] == 0:
            raise ValueError("`actions` must contain at least one step")

        if self.ensembled_actions is None:
            self._ensure_weights(actions_array.shape[0])
            self.ensembled_actions = actions_array.copy()
            self.ensembled_actions_count = np.ones(actions_array.shape[0], dtype=np.int64)
        else:
            assert self.ensembled_actions_count is not None
            overlap = min(self.ensembled_actions.shape[0], actions_array.shape[0])
            max_required_weight = max(
                actions_array.shape[0],
                int(self.ensembled_actions_count[:overlap].max(initial=0)) + 1 if overlap > 0 else 1,
            )
            self._ensure_weights(max_required_weight)
            assert self._ensemble_weights is not None
            assert self._ensemble_weights_cumsum is not None

            if overlap > 0:
                overlap_counts = self.ensembled_actions_count[:overlap]
                old_weight_sum = self._ensemble_weights_cumsum[overlap_counts - 1]
                new_weight = self._ensemble_weights[overlap_counts]
                new_weight_sum = self._ensemble_weights_cumsum[overlap_counts]

                self.ensembled_actions[:overlap] = (
                    self.ensembled_actions[:overlap] * old_weight_sum[:, None]
                    + actions_array[:overlap] * new_weight[:, None]
                ) / new_weight_sum[:, None]
                self.ensembled_actions_count[:overlap] = overlap_counts + 1

            if actions_array.shape[0] > overlap:
                self.ensembled_actions = np.concatenate(
                    [self.ensembled_actions, actions_array[overlap:]],
                    axis=0,
                )
                self.ensembled_actions_count = np.concatenate(
                    [
                        self.ensembled_actions_count,
                        np.ones(actions_array.shape[0] - overlap, dtype=np.int64),
                    ],
                    axis=0,
                )

        assert self.ensembled_actions is not None
        assert self.ensembled_actions_count is not None
        action = self.ensembled_actions[0].copy()
        self.ensembled_actions = self.ensembled_actions[1:]
        self.ensembled_actions_count = self.ensembled_actions_count[1:]
        return action


# ==================== Action Queue Manager (LeRobot Pattern) ====================

class TimestampedActionQueue:
    """
    LeRobot-style action queue with timestamp alignment.

    Key differences from simple deque:
    1. Actions are indexed by timestep, not FIFO order
    2. Newer chunk predictions overwrite overlapping queued actions
    3. Time-based action retrieval (skip expired actions)
    4. Thread-safe operations

    Performance optimization: maintains a sorted timestep list using bisect
    to avoid O(n log n) sorting on every get_action_for_time() call.
    """

    def __init__(self, config: SmoothingConfig):
        self.config = config
        self._queue: Dict[int, TimedAction] = {}  # timestep -> TimedAction
        self._sorted_timesteps: List[int] = []  # Sorted list of timesteps for O(log n) lookup
        self._lock = threading.Lock()
        self._latest_executed_timestep: int = -1
        self._chunk_size: int = 1
    
    def reset(self):
        """Reset queue state for new episode."""
        with self._lock:
            self._queue.clear()
            self._sorted_timesteps.clear()
            self._latest_executed_timestep = -1
    
    def set_chunk_size(self, size: int):
        """Set expected chunk size (for threshold calculation)."""
        self._chunk_size = max(1, size)
    
    def get_queue_size(self) -> int:
        """Get number of actions in queue."""
        with self._lock:
            return len(self._queue)
    
    def get_fill_ratio(self) -> float:
        """Get queue fill ratio relative to chunk size."""
        with self._lock:
            return len(self._queue) / max(1, self._chunk_size)
    
    def should_request_new_chunk(self) -> bool:
        """
        Determine if we should trigger new inference.
        """
        return self.get_fill_ratio() <= self.config.chunk_size_threshold
    
    def add_action_chunk(self, timed_actions: List[TimedAction]):
        """
        Add new action chunk, replacing overlapping future timesteps.

        LeRobot pattern (from robot_client.py _aggregate_action_queues):
        - Skip actions older than latest executed
        - Replace overlapping timesteps with the latest prediction
        - Add new timesteps directly
        """
        import bisect

        with self._lock:
            for new_action in timed_actions:
                timestep = new_action.get_timestep()

                # Skip actions older than what we've already executed
                if timestep <= self._latest_executed_timestep:
                    continue

                # Check if this timestep already exists
                if timestep in self._queue:
                    self._queue[timestep] = new_action
                else:
                    # Add new action directly and maintain sorted timesteps
                    self._queue[timestep] = new_action
                    bisect.insort(self._sorted_timesteps, timestep)

            logger.debug(f"Queue updated: {len(self._queue)} actions, "
                        f"latest_executed={self._latest_executed_timestep}")
    
    def get_action_for_time(self, current_time: float, t0: float) -> Optional[TimedAction]:
        """
        Get action for current time using timestamp alignment.

        LeRobot pattern: calculate which timestep we SHOULD be at,
        then get that action (or nearest future one).

        Args:
            current_time: Current wall clock time
            t0: Episode start time

        Returns:
            TimedAction for current timestep, or None if queue empty
        """
        import bisect

        with self._lock:
            if not self._queue:
                return None

            # Calculate expected timestep based on elapsed time
            elapsed = current_time - t0
            expected_timestep = int(elapsed / self.config.environment_dt)

            # Find the action to execute using binary search on sorted timesteps:
            # 1. If expected_timestep exists, use it
            # 2. If not, use the smallest timestep > latest_executed
            # 3. Skip any timesteps < expected_timestep (they're expired)

            # Use binary search to find first timestep > latest_executed
            idx = bisect.bisect_right(self._sorted_timesteps, self._latest_executed_timestep)

            if idx >= len(self._sorted_timesteps):
                return None

            # Get valid timesteps (those after latest_executed)
            valid_timesteps = self._sorted_timesteps[idx:]

            if not valid_timesteps:
                return None

            # Try to find expected_timestep or the next available one
            target_timestep = None
            for ts in valid_timesteps:
                if ts >= expected_timestep:
                    target_timestep = ts
                    break

            # If all valid timesteps are before expected, use the latest one
            # (this means we're behind, but at least we're moving forward)
            if target_timestep is None:
                target_timestep = valid_timesteps[-1]

            # Get action and update state
            action = self._queue.pop(target_timestep)
            self._sorted_timesteps.remove(target_timestep)
            self._latest_executed_timestep = target_timestep

            # Clean up any expired actions we skipped
            expired = [ts for ts in list(self._sorted_timesteps) if ts < target_timestep]
            for ts in expired:
                del self._queue[ts]
                self._sorted_timesteps.remove(ts)
                logger.debug(f"Discarded expired action timestep {ts}")

            return action
    
    def get_next_action(self) -> Optional[TimedAction]:
        """
        Simple FIFO-style get (fallback when not using timestamp alignment).
        Gets the action with smallest timestep > latest_executed.
        """
        import bisect

        with self._lock:
            if not self._queue:
                return None

            # Use binary search to find first timestep > latest_executed
            idx = bisect.bisect_right(self._sorted_timesteps, self._latest_executed_timestep)

            if idx >= len(self._sorted_timesteps):
                return None

            target_timestep = self._sorted_timesteps[idx]
            action = self._queue.pop(target_timestep)
            self._sorted_timesteps.remove(target_timestep)
            self._latest_executed_timestep = target_timestep

            return action


# ==================== Observation Queue (LeRobot Pattern) ====================

class ObservationQueue:
    """
    LeRobot-style observation queue with maxsize=1.
    
    Key insight: GPU inference is slower than camera capture.
    If we queue observations, we're always processing stale data.
    Solution: Only keep the LATEST observation, discard old ones.
    """
    
    def __init__(self, maxsize: int = 1):
        self._queue: Queue = Queue(maxsize=maxsize)
        self._lock = threading.Lock()
    
    def put(self, obs: TimedObservation) -> bool:
        """
        Add observation, discarding old one if queue is full.
        
        LeRobot pattern (from policy_server.py _enqueue_observation):
        If queue is full, pop the old observation to make room.
        """
        with self._lock:
            if self._queue.full():
                try:
                    _ = self._queue.get_nowait()
                    logger.debug("Observation queue full, discarded oldest")
                except Empty:
                    pass
            
            try:
                self._queue.put_nowait(obs)
                return True
            except Full:
                return False
    
    def get(self, timeout: float = 0.1) -> Optional[TimedObservation]:
        """Get observation with timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_nowait(self) -> Optional[TimedObservation]:
        """Get observation without blocking."""
        try:
            return self._queue.get_nowait()
        except Empty:
            return None
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
    
    def clear(self):
        """Clear all observations."""
        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Empty:
                    break

# ==================== Async Inference Engine ====================

class AsyncInferenceWorker:
    """
    Background inference worker thread.
    
    LeRobot pattern:
    - Observation queue maxsize=1 (always latest)
    - Continuous inference in background
    - Results go to action queue with timestamp
    """
    
    def __init__(
        self,
        config: SmoothingConfig,
        inference_fn: Callable[[Dict[str, np.ndarray], np.ndarray], np.ndarray],
        action_queue: TimestampedActionQueue,
    ):
        self.config = config
        self._inference_fn = inference_fn
        self._action_queue = action_queue
        
        self._obs_queue = ObservationQueue(maxsize=config.obs_queue_maxsize)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._must_go_event = threading.Event()
        self._must_go_event.set()  # Initially set
    
    def start(self):
        """Start background inference thread."""
        if self._running:
            return

        self._running = True

        # Register with thread monitor
        monitor = get_inference_monitor()
        monitor.register_thread(
            name="AsyncInferenceWorker",
            expected_interval=2.0,  # Inference may take time, allow slack
            timeout_threshold=10.0  # Consider dead if no heartbeat for 10s
        )

        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="AsyncInferenceWorker"
        )
        self._thread.start()
        logger.info("Async inference worker started")
    
    def stop(self):
        """Stop background inference thread."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._obs_queue.clear()

        # Unregister from thread monitor
        monitor = get_inference_monitor()
        monitor.unregister_thread("AsyncInferenceWorker")

        logger.info("Async inference worker stopped")
    
    def submit_observation(self, obs: TimedObservation):
        """
        Submit observation for inference.
        
        LeRobot pattern:
        - If queue empty and must_go set, mark observation as must_go
        - must_go observations bypass similarity checks
        """
        # Check if we need this observation processed urgently
        if self._must_go_event.is_set() and self._action_queue.get_queue_size() == 0:
            obs.must_go = True
            self._must_go_event.clear()
        
        self._obs_queue.put(obs)
    
    def _worker_loop(self):
        """Background worker loop."""
        monitor = get_inference_monitor()

        while self._running:
            # Send heartbeat at start of each iteration
            monitor.heartbeat("AsyncInferenceWorker")

            try:
                # Get observation (with timeout to allow clean shutdown)
                obs = self._obs_queue.get(timeout=0.1)
                if obs is None:
                    continue
                
                # Check if we should process this observation
                should_process = self._should_process_observation(obs)
                
                if not should_process:
                    logger.debug(f"Skipping observation timestep {obs.get_timestep()}")
                    continue
                
                # Run inference
                start_time = time.perf_counter()
                action_chunk = self._inference_fn(obs.images, obs.state)
                elapsed = time.perf_counter() - start_time
                
                # Convert to TimedActions
                timed_actions = self._time_action_chunk(
                    t_0=obs.get_timestamp(),
                    action_chunk=action_chunk,
                    i_0=obs.get_timestep()
                )
                
                # Add to action queue
                self._action_queue.add_action_chunk(timed_actions)
                
                # Signal that we've processed - next empty queue triggers must_go
                self._must_go_event.set()
                
                logger.debug(
                    f"queue_size={self._action_queue.get_queue_size()}"
                )
                
            except Exception as e:
                logger.error(f"Async inference error: {e}")
                import traceback
                traceback.print_exc()
    
    def _should_process_observation(self, obs: TimedObservation) -> bool:
        """
        Check if observation should be processed.
        
        LeRobot pattern:
        - must_go observations are always processed
        - Otherwise, check if we need new actions
        """
        if obs.must_go:
            return True
        
        # Check if action queue needs refilling
        return self._action_queue.should_request_new_chunk()
    
    def _time_action_chunk(
        self,
        t_0: float,
        action_chunk: np.ndarray,
        i_0: int
    ) -> List[TimedAction]:
        """
        Convert action chunk to TimedAction list.
        
        LeRobot pattern (from policy_server.py _time_action_chunk):
        First action corresponds to t_0, rest are t_0 + i*dt
        """
        dt = self.config.environment_dt
        return [
            TimedAction(
                timestamp=t_0 + i * dt,
                timestep=i_0 + i,
                action=action_chunk[i]
            )
            for i in range(len(action_chunk))
        ]


# ==================== Base Inference Engine ====================

class BaseInferenceEngine(ABC):
    """
    Abstract base class for inference engines with LeRobot-style async support.
    
    Key Features:
    - Timestamp-aligned action queue
    - Background inference thread
    - Chunk-threshold based refill
    """
    
    def __init__(self, smoothing_config: Optional[SmoothingConfig] = None):
        self.is_loaded = False
        self.model_type: str = ""
        self.required_cameras: List[str] = []
        self.state_dim: int = 0
        self.action_dim: int = 7
        self.chunk_size: int = 1
        self.n_action_steps: int = 1
        self.requested_device: Optional[str] = None
        self.actual_device: Optional[str] = None
        self.device_warning: str = ""
        
        # Config
        self.smoothing_config = smoothing_config or SmoothingConfig()
        
        # Components (initialized after model load)
        self._action_queue: Optional[TimestampedActionQueue] = None
        self._async_worker: Optional[AsyncInferenceWorker] = None
        self._temporal_ensembler: Optional[TemporalEnsembler] = None
        self._last_action: Optional[np.ndarray] = None
        
        # Episode state
        self._episode_start_time: float = 0.0
        self._current_timestep: int = 0
        self._fallback_count: int = 0
    
    def _init_components(self):
        """Initialize all components after model is loaded."""
        if self.smoothing_config.temporal_ensemble_enabled and self.smoothing_config.enable_async_inference:
            logger.warning(
                "%s temporal ensembling requires per-step inference; disabling async inference.",
                self.model_type or "Inference",
            )
            self.smoothing_config.enable_async_inference = False

        if self.smoothing_config.enable_async_inference and self.n_action_steps <= 1:
            logger.warning(
                "%s model reports n_action_steps=%s; disabling async inference because a single-step policy "
                "cannot keep a 30Hz action queue filled.",
                self.model_type or "Inference",
                self.n_action_steps,
            )
            self.smoothing_config.enable_async_inference = False

        self._action_queue = TimestampedActionQueue(self.smoothing_config)
        self._action_queue.set_chunk_size(self.n_action_steps)

        self._temporal_ensembler = None
        if self.smoothing_config.temporal_ensemble_enabled:
            assert self.smoothing_config.temporal_ensemble_coeff is not None
            self._temporal_ensembler = TemporalEnsembler(self.smoothing_config.temporal_ensemble_coeff)

        if self.smoothing_config.enable_async_inference:
            self._async_worker = AsyncInferenceWorker(
                config=self.smoothing_config,
                inference_fn=self._predict_chunk,
                action_queue=self._action_queue,
            )

    def _require_loaded(self):
        if not self.is_loaded:
            raise RuntimeError("Policy is not loaded")

    def _require_runtime_ready(self):
        if self._action_queue is None:
            raise RuntimeError(
                "Policy runtime is not initialized. Load the policy before running inference."
            )

    def _validate_images_state(self, images: Dict[str, np.ndarray], state: np.ndarray):
        if not isinstance(images, dict):
            raise ValueError("`images` must be a dict of camera_role -> numpy.ndarray")
        if not images:
            raise ValueError("`images` cannot be empty")

        for camera_role, image in images.items():
            if not isinstance(camera_role, str) or not camera_role.strip():
                raise ValueError("Image keys must be non-empty strings")
            if not isinstance(image, np.ndarray):
                raise ValueError(f"Image `{camera_role}` must be a numpy.ndarray")
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"Image `{camera_role}` must have shape (H, W, 3), got {tuple(image.shape)}"
                )

        if not isinstance(state, np.ndarray):
            raise ValueError("`state` must be a numpy.ndarray")
        if state.ndim != 1:
            raise ValueError(f"`state` must be a 1D numpy.ndarray, got ndim={state.ndim}")

    def _validate_observation(self, observation: Observation):
        if not isinstance(observation, Observation):
            raise ValueError("`observation` must be an inference_sdk.Observation instance")
        self._validate_images_state(observation.images, observation.state)
    
    @abstractmethod
    def load(self, checkpoint_dir: str) -> Tuple[bool, str]:
        """Load model from checkpoint directory."""
        pass

    @abstractmethod
    def build_inference_frame(self, images: Dict[str, np.ndarray], state: np.ndarray) -> Dict[str, Any]:
        """Build the model input frame from raw SDK observations."""
        pass
    
    @abstractmethod
    def _predict_chunk(self, images: Dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        """
        Predict action chunk from observation.
        
        Args:
            images: Dict of {camera_role: image array (H, W, 3)}
            state: Robot state array (state_dim,)
            
        Returns:
            Action chunk (n_action_steps, action_dim)
        """
        pass
    
    def reset(self):
        """Reset state for new episode."""
        # Stop async worker if running
        if self._async_worker is not None:
            self._async_worker.stop()
        
        # Reset all components
        if self._action_queue is not None:
            self._action_queue.reset()
        if self._temporal_ensembler is not None:
            self._temporal_ensembler.reset()
        
        # Reset episode state
        self._episode_start_time = time.time()
        self._current_timestep = 0
        self._fallback_count = 0
        self._last_action = None
        
        logger.debug(f"{self.model_type} inference engine reset")
    
    def start_async_inference(self):
        """Start background inference thread."""
        if self._async_worker is not None:
            self._async_worker.start()
    
    def stop_async_inference(self):
        """Stop background inference thread."""
        if self._async_worker is not None:
            self._async_worker.stop()

    def _reset_runtime_buffers(self):
        """Clear pending scheduling state without resetting episode timing."""
        if self._action_queue is not None:
            self._action_queue.reset()
        if self._temporal_ensembler is not None:
            self._temporal_ensembler.reset()
    
    def select_action(self, images: Dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        """
        Select action with LeRobot-style timestamp alignment.
        
        Flow:
        1. Create TimedObservation and submit to async worker
        2. Get action from queue (timestamp-aligned or fallback)
        3. Cache last action for fallback
        4. Increment timestep
        """
        self._require_loaded()
        self._require_runtime_ready()
        self._validate_images_state(images, state)
        
        current_time = time.time()
        
        # Sync timestep with wall clock to handle loop lag
        # If loop is slower than control_fps, we need to skip timesteps to stay aligned
        elapsed = max(0.0, current_time - self._episode_start_time)
        self._current_timestep = int(elapsed / self.smoothing_config.environment_dt)
        
        # Generic temporal ensembling: run inference every step and blend
        # overlapping future actions online. This intentionally bypasses the
        # action queue because queue caching would defeat per-step re-planning.
        if self._temporal_ensembler is not None:
            start_time = time.perf_counter()
            action_chunk = self._predict_chunk(images, state)
            elapsed = time.perf_counter() - start_time
            action = self._temporal_ensembler.update(action_chunk)
            self._last_action = action.copy()
            logger.debug(f"Temporal ensemble inference: {elapsed*1000:.1f}ms")
            return action

        # Create timed observation
        obs = TimedObservation(
            timestamp=current_time,
            timestep=self._current_timestep,
            images=images,
            state=state,
            must_go=False
        )
        
        # Submit to async worker (if enabled)
        if self._async_worker is not None and self._async_worker._running:
            self._async_worker.submit_observation(obs)
            
            # Get action from queue with timestamp alignment
            timed_action = self._action_queue.get_action_for_time(
                current_time, 
                self._episode_start_time
            )
        else:
            # Synchronous mode: run inference directly if queue empty
            timed_action = self._action_queue.get_next_action()
            
            if timed_action is None:
                # Run synchronous inference
                start_time = time.perf_counter()
                action_chunk = self._predict_chunk(images, state)
                elapsed = time.perf_counter() - start_time

                # Add to queue
                timed_actions = [
                    TimedAction(
                        timestamp=current_time + i * self.smoothing_config.environment_dt,
                        timestep=self._current_timestep + i,
                        action=action_chunk[i]
                    )
                    for i in range(len(action_chunk))
                ]
                self._action_queue.add_action_chunk(timed_actions)
                
                # Get first action
                timed_action = self._action_queue.get_next_action()
                
                logger.debug(f"Sync inference: {elapsed*1000:.1f}ms")
        
        # Handle empty queue (fallback)
        if timed_action is None:
            action = self._get_fallback_action(state)
            self._fallback_count += 1
            logger.debug(f"Using fallback action (count={self._fallback_count})")
        else:
            action = timed_action.get_action()
        self._last_action = action.copy()
        
        # Timestep is updated at start of method based on wall clock
        # self._current_timestep += 1
        
        return action

    def _maybe_apply_instruction(self, instruction: Optional[str]):
        """Update the current instruction when the SDK caller provides one."""
        if instruction is None:
            return

        get_instruction = getattr(self, "get_instruction", None)
        if callable(get_instruction):
            try:
                if get_instruction() == instruction:
                    return
            except Exception:
                logger.debug("Failed to read current instruction", exc_info=True)

        set_instruction = getattr(self, "set_instruction", None)
        if callable(set_instruction):
            if not set_instruction(instruction):
                raise RuntimeError(f"Failed to set instruction: {instruction}")
            self._reset_runtime_buffers()
            return

        raise RuntimeError(
            f"{self.model_type or 'Policy'} does not support language instructions"
        )

    def step(self, observation: Observation) -> np.ndarray:
        """
        SDK-friendly single-step API.

        This wraps the existing queue-aware `select_action()` behavior and optionally
        updates the language instruction when provided by the caller.
        """
        self._require_loaded()
        self._validate_observation(observation)
        self._maybe_apply_instruction(observation.instruction)
        return self.select_action(observation.images, observation.state)

    def predict_chunk(self, observation: Observation) -> np.ndarray:
        """
        SDK-friendly raw chunk prediction API without queue scheduling.
        """
        self._require_loaded()
        self._validate_observation(observation)
        self._maybe_apply_instruction(observation.instruction)
        return self._predict_chunk(observation.images, observation.state)
    
    def _get_fallback_action(self, state: np.ndarray) -> np.ndarray:
        """Get fallback action when queue is empty."""
        mode = self.smoothing_config.fallback_mode
        
        if mode == "repeat" and self._last_action is not None:
            return self._last_action.copy()
        
        # "hold" mode or no last action: return current state
        return state[:self.action_dim].copy() if len(state) >= self.action_dim else np.zeros(self.action_dim)
    
    # ==================== Status Methods ====================
    
    def get_queue_size(self) -> int:
        """Get current action queue size."""
        if self._action_queue is not None:
            return self._action_queue.get_queue_size()
        return 0
    
    def get_fallback_count(self) -> int:
        """Get count of fallback uses."""
        return self._fallback_count
    
    def get_required_cameras(self) -> List[str]:
        """Return list of required camera roles."""
        return self.required_cameras
    
    def get_state_dim(self) -> int:
        """Return expected state dimension."""
        return self.state_dim

    def get_device_status(self) -> Dict[str, Optional[str]]:
        """Return requested/actual device metadata for observability."""
        return {
            "requested_device": self.requested_device,
            "actual_device": self.actual_device,
            "device_warning": self.device_warning,
        }

    def get_metadata(self) -> PolicyMetadata:
        """Return static policy metadata for SDK consumers."""
        return PolicyMetadata(
            model_type=self.model_type,
            required_cameras=list(self.required_cameras),
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            chunk_size=self.chunk_size,
            n_action_steps=self.n_action_steps,
            requested_device=self.requested_device,
            actual_device=self.actual_device,
            extras={
                "async_inference_enabled": bool(
                    getattr(self, "smoothing_config", None)
                    and self.smoothing_config.enable_async_inference
                ),
                "temporal_ensemble_enabled": bool(
                    getattr(self, "smoothing_config", None)
                    and self.smoothing_config.temporal_ensemble_enabled
                ),
                "temporal_ensemble_coeff": (
                    self.smoothing_config.temporal_ensemble_coeff
                    if getattr(self, "smoothing_config", None) is not None
                    else None
                ),
            },
        )

    def get_status(self) -> PolicyStatus:
        """Return queue/runtime status for SDK consumers."""
        return PolicyStatus(
            is_loaded=bool(self.is_loaded),
            model_type=self.model_type,
            queue_size=self.get_queue_size(),
            fallback_count=self.get_fallback_count(),
            required_cameras=list(self.get_required_cameras()),
            requested_device=self.requested_device,
            actual_device=self.actual_device,
            device_warning=self.device_warning,
            async_inference_enabled=bool(
                getattr(self, "smoothing_config", None)
                and self.smoothing_config.enable_async_inference
            ),
            temporal_ensemble_enabled=bool(
                getattr(self, "smoothing_config", None)
                and self.smoothing_config.temporal_ensemble_enabled
            ),
        )
    
    def set_control_fps(self, fps: float):
        """Update control frequency."""
        self.smoothing_config.control_fps = fps
        if self._action_queue is not None:
            self._action_queue.config.control_fps = fps
    
    def set_smoothing_config(self, config: SmoothingConfig):
        """Update smoothing configuration."""
        self.smoothing_config = config
    
    @staticmethod
    def validate_checkpoint(checkpoint_dir: str) -> Tuple[bool, str]:
        """Validate that checkpoint directory contains required files."""
        path = Path(checkpoint_dir)
        
        if not path.exists():
            return False, f"Checkpoint目录不存在: {checkpoint_dir}"
        
        required_files = ["inference_config.yaml", "model.pth", "stats.json"]
        missing = []
        for f in required_files:
            if not (path / f).exists():
                missing.append(f)
        
        if missing:
            return False, f"缺少必需文件: {', '.join(missing)}"
        
        return True, ""
    
    @abstractmethod
    def unload(self):
        """Unload model and free memory."""
        pass

    def close(self):
        """Alias for unload() to match common SDK usage patterns."""
        self.unload()

    def __enter__(self):
        self._require_loaded()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
