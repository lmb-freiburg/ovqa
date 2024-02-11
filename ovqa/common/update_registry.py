from ovqa import models, tasks, processors, runners, datasets


def update_registry():
    _ = (models, tasks, processors, runners, datasets)
