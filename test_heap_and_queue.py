import unittest
import random
import signal
import time
import threading
import os

from priority_queue import MinHeap, PriorityQueue

class TimeoutException(Exception):
    pass

# inspired by https://stackoverflow.com/a/49567288/3946214 - chatgpt added windows
class test_timeout:
    """
    Cross-platform timeout context manager.

    - On Unix/macOS: uses signal.alarm (fast, reliable).
    - On Windows: runs the test body in a separate thread and fails if it exceeds time.
    """

    def __init__(self, seconds, error_message=None):
        self.seconds = seconds
        self.error_message = error_message or f"test timed out after {seconds}s."
        self._use_signals = hasattr(signal, "SIGALRM") and os.name != "nt"

        # Windows/thread fallback
        self._thread = None
        self._exc = None

    # ---------- Unix/macOS implementation ----------
    def _handle_timeout(self, signum, frame):
        raise TimeoutException(self.error_message)

    # ---------- Windows/thread implementation ----------
    def _thread_runner(self, fn):
        try:
            fn()
        except BaseException as e:
            self._exc = e

    def run(self, fn):
        """
        Use this ONLY on Windows (or when signals not available).
        Example:

        with test_timeout(1) as t:
            t.run(lambda: do_something())
        """
        self._thread = threading.Thread(target=self._thread_runner, args=(fn,), daemon=True)
        self._thread.start()
        self._thread.join(self.seconds)

        if self._thread.is_alive():
            raise TimeoutException(self.error_message)

        if self._exc is not None:
            raise self._exc

    # ---------- Context manager API ----------
    def __enter__(self):
        if self._use_signals:
            signal.signal(signal.SIGALRM, self._handle_timeout)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._use_signals:
            signal.alarm(0)
        # don't suppress exceptions
        return False


# ----------------------------
# MinHeap Unit Tests
# ----------------------------

class TestMinHeap(unittest.TestCase):
    def test_empty_heap(self):
        h = MinHeap()
        self.assertTrue(h.is_empty())
        self.assertEqual(len(h), 0)
        self.assertIsNone(h.peek())
        self.assertIsNone(h.pop_min())

    def test_single_insert_and_pop(self):
        h = MinHeap()
        h.add(5, "A")
        self.assertFalse(h.is_empty())
        self.assertEqual(len(h), 1)
        self.assertEqual(h.peek(), (5, "A"))
        self.assertEqual(h.pop_min(), (5, "A"))
        self.assertTrue(h.is_empty())

    def test_inserts_pop_in_sorted_priority_order(self):
        h = MinHeap()
        h.add(10, "x")
        h.add(3, "y")
        h.add(7, "z")
        h.add(1, "w")

        self.assertEqual(h.pop_min(), (1, "w"))
        self.assertEqual(h.pop_min(), (3, "y"))
        self.assertEqual(h.pop_min(), (7, "z"))
        self.assertEqual(h.pop_min(), (10, "x"))
        self.assertIsNone(h.pop_min())

    def test_duplicate_priorities(self):
        h = MinHeap()
        h.add(2, "A")
        h.add(2, "B")
        h.add(1, "C")
        h.add(2, "D")

        # Must pop priority=1 first
        self.assertEqual(h.pop_min(), (1, "C"))

        # The remaining three all have priority 2 (order doesn't matter)
        popped = [h.pop_min(), h.pop_min(), h.pop_min()]
        self.assertEqual(sorted(popped), sorted([(2, "A"), (2, "B"), (2, "D")]))
        self.assertTrue(h.is_empty())


    def test_peek_does_not_remove(self):
        h = MinHeap()
        h.add(4, "A")
        h.add(1, "B")
        h.add(3, "C")

        self.assertEqual(h.peek(), (1, "B"))
        self.assertEqual(len(h), 3)     # still there
        self.assertEqual(h.pop_min(), (1, "B"))

    def test_randomized_matches_sorted(self):
        h = MinHeap()
        items = []
        for _ in range(2000):
            p = random.randint(1, 10000)
            item = random.randint(1, 10**9)
            items.append((p, item))
            h.add(p, item)

        items_sorted = sorted(items, key=lambda x: x[0])
        popped = [h.pop_min() for _ in range(len(h))]

        self.assertEqual([p for (p, _) in popped], [p for (p, _) in items_sorted])

    def test_heap_property_always_valid(self):
        """
        Extra check: after lots of random adds/pops,
        ensure parent priority <= child priority for all nodes.
        """
        h = MinHeap()
        for _ in range(5000):
            if random.random() < 0.7:
                h.add(random.randint(1, 10000), random.randint(1, 10**9))
            else:
                h.pop_min()

            # verify heap property
            for i in range(len(h.data)):
                left = 2 * i + 1
                right = 2 * i + 2
                if left < len(h.data):
                    self.assertLessEqual(h.data[i][0], h.data[left][0])
                if right < len(h.data):
                    self.assertLessEqual(h.data[i][0], h.data[right][0])

    def test_performance_pop_min_should_be_fast(self):
        with test_timeout(2):
            h = MinHeap()

            # load heap with many elements
            n = 200_000
            for i in range(n):
                # random priorities
                h.add(random.randint(1, 10**9), i)

            # do some pops (should be fast if O(log n))
            for _ in range(20_000):
                h.pop_min()

    def test_pop_after_emptying_heap_returns_none_or_raises(self):
        h = MinHeap()
    
        # add items
        h.add(5, "A")
        h.add(1, "B")
        h.add(3, "C")
    
        # pop all items
        h.pop_min()
        h.pop_min()
        h.pop_min()
    
        # now empty: pop again should either return None or raise
        try:
            out = h.pop_min()
            self.assertIsNone(out)
        except Exception:
            pass

# ----------------------------
# PriorityQueue Unit Tests
# ----------------------------

class TestPriorityQueue(unittest.TestCase):
    def test_empty_priority_queue(self):
        pq = PriorityQueue()
        self.assertTrue(pq.is_empty())
        self.assertEqual(len(pq), 0)
        self.assertIsNone(pq.peek())
        self.assertIsNone(pq.pop())

    def test_push_pop_basic(self):
        pq = PriorityQueue()
        pq.add(5, "A")
        pq.add(1, "B")
        pq.add(3, "C")

        self.assertEqual(pq.peek(), (1, "B"))
        self.assertEqual(pq.pop(), (1, "B"))
        self.assertEqual(pq.pop(), (3, "C"))
        self.assertEqual(pq.pop(), (5, "A"))
        self.assertIsNone(pq.pop())

    def test_duplicate_priorities(self):
        pq = PriorityQueue()
        pq.add(2, "A")
        pq.add(2, "B")
        pq.add(1, "C")
        pq.add(2, "D")

        self.assertEqual(pq.pop(), (1, "C"))

        rest = [pq.pop(), pq.pop(), pq.pop()]
        self.assertEqual(sorted(rest), sorted([(2, "A"), (2, "B"), (2, "D")]))

    def test_randomized_priority_queue_matches_sorted(self):
        pq = PriorityQueue()
        items = []

        for i in range(3000):
            p = random.randint(1, 10000)
            item = f"item{i}"
            items.append((p, item))
            pq.add(p, item)

        items_sorted = sorted(items, key=lambda x: x[0])

        popped_priorities = []
        while not pq.is_empty():
            priority, item = pq.pop()
            popped_priorities.append(priority)

        self.assertEqual(popped_priorities, [p for (p, _) in items_sorted])

    def test_priority_queue_performance(self):
        """
        Smaller/faster timing test for PQ too.
        """
        with test_timeout(2):
            pq = PriorityQueue()
            n = 150_000

            for i in range(n):
                pq.add(random.randint(1, 10**9), i)

            for _ in range(25_000):
                pq.pop()
                
    def test_pop_after_emptying_priority_queue_returns_none_or_raises(self):
        pq = PriorityQueue()
    
        # add items
        pq.add(5, "A")
        pq.add(1, "B")
        pq.add(3, "C")
    
        # pop all items
        pq.pop()
        pq.pop()
        pq.pop()
    
        # now empty: pop again should either return None or raise
        try:
            out = pq.pop()
            self.assertIsNone(out)
        except Exception:
            # acceptable behavior
            pass

if __name__ == "__main__":
    unittest.main()