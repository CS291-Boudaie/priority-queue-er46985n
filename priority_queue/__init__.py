from .priority_queue import MinHeap, PriorityQueue  # template defaults

try:
    from .answers import MinHeap, PriorityQueue  # overrides locally
except ImportError:
    pass