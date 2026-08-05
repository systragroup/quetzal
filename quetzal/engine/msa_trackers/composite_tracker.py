from quetzal.engine.msa_trackers.tracker import Tracker


class CompositeTracker(Tracker):
    # this tracker is used to pass a list of tracker to the road_pathfinder.
    def __init__(self, tracker_list: list[Tracker]):
        self.trackers = tracker_list

    def init(self, *args, **kwargs):
        for tracker in self.trackers:
            if tracker():
                tracker.init(*args, **kwargs)

    def __call__(self) -> bool:
        return any(tracker() for tracker in self.trackers)

    def assign(self, *args, **kwargs):
        for tracker in self.trackers:
            if tracker():
                tracker.assign(*args, **kwargs)

    def add_weights(self, *args, **kwargs):
        for tracker in self.trackers:
            if tracker():
                tracker.add_weights(*args, **kwargs)

    def merge(self, *args, **kwargs) -> list:
        return [tracker.merge(*args, **kwargs) for tracker in self.trackers]
