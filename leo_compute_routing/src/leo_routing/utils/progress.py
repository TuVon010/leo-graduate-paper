"""Wall-clock progress; FPS counts completed environment slots, not tasks."""
import math
from time import perf_counter


def duration_text(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "NA"
    seconds = max(0, seconds)
    if seconds < 60:
        return "%.2fs" % seconds
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "%02d:%02d:%02d" % (hours, minutes, seconds)


def metric_text(value, precision=4):
    return "NA" if value is None else ("%.*f" % (precision, value))


def device_summary(device, requested=None):
    import torch
    using_gpu = device.type == "cuda"
    available = torch.cuda.is_available()
    index = (device.index if device.index is not None else torch.cuda.current_device()) if using_gpu else None
    text = "requested=%s actual=%s | CUDA_available=%s GPU_used=%s" % (
        requested or device, ("cuda:%s" % index) if using_gpu else device,
        "yes" if available else "no", "yes" if using_gpu else "no")
    if using_gpu:
        text += " | GPU=%s VRAM_total=%.0f MiB" % (
            torch.cuda.get_device_name(index), torch.cuda.get_device_properties(index).total_memory / 2**20)
    else:
        text += " | model=CPU"
    return text + " | torch_threads=%s" % torch.get_num_threads()


def gpu_memory(device):
    if device.type != "cuda":
        return {}
    import torch
    return {"gpu_allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
            "gpu_reserved_mib": torch.cuda.memory_reserved(device) / 2**20}


class EpisodeProgress:
    def __init__(self, config, callback=None, label="ROLLOUT", interval_seconds=10.0):
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("Log interval must be positive and finite")
        self.callback, self.label, self.interval = callback, label, interval_seconds
        self.started = self.last_report = perf_counter()
        self.admission_slots = config["simulation"]["slots"]
        self.max_slots = self.admission_slots + config["simulation"]["drain_slots"]
        self.slot_seconds = config["simulation"]["slot_seconds"]
        self.steps, self.tasks, self.reward = 0, 0, 0.0

    def step(self, info, reward):
        self.steps += 1
        self.tasks += info["slot_metrics"]["arrivals"]
        self.reward += reward
        now = perf_counter()
        if self.callback and (self.steps == 1 or now - self.last_report >= self.interval):
            self.last_report = now
            elapsed = now - self.started
            self.callback("[%s] slot=%s/%s(max) phase=%s sim=%.2fs | elapsed=%s FPS=%.2f tasks/s=%.2f "
                          "admitted=%s active=%s reward_so_far=%.4f" % (
                              self.label, self.steps, self.max_slots,
                              "admission" if self.steps <= self.admission_slots else "drain",
                              self.steps * self.slot_seconds, duration_text(elapsed), self.steps / max(elapsed, 1e-9),
                              self.tasks / max(elapsed, 1e-9), self.tasks, info["slot_metrics"]["active_tasks"], self.reward))

    def metrics(self):
        elapsed = perf_counter() - self.started
        return {"environment_steps": self.steps, "episode_wall_seconds": elapsed,
                "environment_fps": self.steps / max(elapsed, 1e-9),
                "admitted_tasks_per_second": self.tasks / max(elapsed, 1e-9)}


def outcome_text(metrics):
    return ("reward=%s success=%.2f%% delay=%ss P95=%ss ddl=%.2f%% route_fail=%.2f%% route_reject=%.2f%% censor=%.2f%% "
            "task_cost=%ss sat_CPU_util=%.2f%% link_util=%.2f%% completed=%s/%s" % (
        metric_text(metrics["total_reward"]), 100 * metrics["success_rate"],
        metric_text(metrics["mean_completion_delay_s"]), metric_text(metrics["p95_completion_delay_s"]),
        100 * metrics["deadline_violation_rate"], 100 * metrics["route_failure_rate"],
        100 * metrics.get("routing_rejection_rate", 0), 100 * metrics["censored_rate"],
        metric_text(metrics.get("mean_cost_per_admitted_task_s")), 100 * metrics["cpu_utilization"],
        100 * metrics["link_utilization"], metrics["completed_count"], metrics["task_count"]))
