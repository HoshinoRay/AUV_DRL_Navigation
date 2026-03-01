from .navigation_task import NavigationTask
from .avoidance_task import AvoidanceTask
from .domain_navigation_task import DomainNavigationTask

# 注册表：通过字符串名字找到类
TASK_REGISTRY = {
    "navigate": NavigationTask,
    "avoidance": AvoidanceTask,
    'domain_navigation': DomainNavigationTask
}