"""Event-level cuts and a tiny counter for monitoring rejection rates."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FailCounter:
    n_events: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    # Ordered list of stages, in the order events are rejected.
    stages: list[str] = field(default_factory=list)

    def bump(self, stage: str) -> None:
        if stage not in self.counts:
            self.counts[stage] = 0
            self.stages.append(stage)
        self.counts[stage] += 1

    def report(self) -> None:
        events = self.n_events
        for stage in self.stages:
            n = self.counts[stage]
            if events <= 0:
                print(f"Percentage of events cut due to {stage}: n/a (no events left)")
            else:
                print(f"Percentage of events cut due to {stage}: {100 * n / events:.2f}%")
            events -= n
        print(f"Total events processed: {self.n_events}")
        print(f"Surviving number of events: {events}")


PT_FATJET_MIN = 250.0
ETA_FATJET_MAX = 3.0

PHOTON_PT_MIN = 20.0
PHOTON_ETA_MAX = 2.5
PHOTON_PT_OVER_MGG_LEAD = 0.35
PHOTON_PT_OVER_MGG_SUB = 0.25
MGG_MIN = 110.0
MGG_MAX = 140.0
