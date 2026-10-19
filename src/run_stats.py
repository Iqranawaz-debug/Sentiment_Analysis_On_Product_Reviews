"""Wall-clock, CPU time and OS process high-water resident memory."""

import os
import sys
import time


def peak_rss_mb():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise OSError(ctypes.get_last_error(), "Cannot measure peak RSS")
        return counters.PeakWorkingSetSize / (1024**2)
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if sys.platform == "darwin" else 1024)


class RunStats:
    def __init__(self):
        self.wall = time.perf_counter()
        self.cpu = time.process_time()

    def snapshot(self):
        return {
            "training_seconds": time.perf_counter() - self.wall,
            "cpu_seconds": time.process_time() - self.cpu,
            "peak_memory_mb": peak_rss_mb(),
            "memory_scope": "OS process peak RSS since process startup; excludes child processes",
            "pid": os.getpid(),
        }
