"""No-learning control for the completion-time prior, shield and prefix calendar."""
import numpy as np

from ..agents.features import FeatureBuilder, CONTACT_COMPLETION_INDEX
from ..agents.settings import rl_settings, configure_rl_environment
from ..routing.reservations import ReservationCalendar


class ContactGreedy:
    name = "contact_greedy"
    use_future = True

    def __init__(self, config):
        if config is None:
            raise ValueError("contact_greedy requires the experiment configuration")
        self.config = config
        self.settings = rl_settings(config)
        self.settings.update(use_mask=True, shield_mode="contact", use_reservations=True)
        self.features = FeatureBuilder(config, self.settings)

    def configure_environment(self, config):
        configured = configure_rl_environment(config, self.settings)
        configured["routing"]["candidate_generation"] = "contact"
        return configured

    def select(self, observation):
        actions = {}
        cpu, links = np.zeros_like(observation.cpu_capacities), {}
        calendar = ReservationCalendar.from_observation(observation)
        for position, task in enumerate(sorted(observation.tasks, key=lambda t: (t.deadline_seconds, t.task_id))):
            decision = self.features.decision_input(observation, task, position, cpu, links, calendar)
            # log1p is strictly increasing: same ranking as physical completion time.
            scores = decision.candidates[:, CONTACT_COMPLETION_INDEX]
            allowed = np.flatnonzero(decision.mask)
            choice = int(allowed[np.argmin(scores[allowed])])
            actions[task.task_id] = choice
            self.features.book(observation.candidates[task.task_id][choice].action, task, cpu, links, calendar)
        return actions
