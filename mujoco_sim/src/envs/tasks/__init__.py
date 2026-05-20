from .navigation_task import NavigationTask
from .avoidance_task import AvoidanceTask
from .domain_navigation_task import DomainNavigationTask

TASK_REGISTRY = {
    "navigate": NavigationTask,
    "avoidance": AvoidanceTask,
    'domain_navigation': DomainNavigationTask
}