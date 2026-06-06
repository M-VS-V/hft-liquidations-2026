import numpy as np

from shaw_features import windows

S = 1_000_000


def test_window_sum_is_half_open():
    ts = np.array([0, 10, 20, 30], dtype=np.int64)
    val = np.array([1.0, 2.0, 3.0, 4.0])
    prefix = windows.build_prefix(val)
    q = np.array([30], dtype=np.int64)
    # [30-25, 30) = [5, 30) -> indices 10, 20 -> 2+3 = 5 ; event at 30 excluded.
    assert windows.window_sum(ts, prefix, q, 25)[0] == 5.0


def test_window_count_excludes_event_at_query():
    ts = np.array([0, 10, 20, 30], dtype=np.int64)
    q = np.array([30], dtype=np.int64)
    assert windows.window_count(ts, q, 100)[0] == 3.0  # 0,10,20 (not 30)


def test_time_since_last():
    ts = np.array([5 * S, 45 * S], dtype=np.int64)
    q = np.array([50 * S], dtype=np.int64)
    assert abs(windows.time_since_last(ts, q)[0] - 5.0) < 1e-9


def test_time_since_last_default_when_empty():
    ts = np.array([], dtype=np.int64)
    q = np.array([10], dtype=np.int64)
    assert windows.time_since_last(ts, q, default_s=1e9)[0] == 1e9


def test_asof_prev_forward_fill_and_future_nan():
    ts = np.array([0, 10, 20], dtype=np.int64)
    val = np.array([1.0, 2.0, 3.0])
    q = np.array([-1, 5, 25], dtype=np.int64)
    out = windows.asof_prev(ts, val, q, allow_future=False)
    assert np.isnan(out[0])  # before first
    assert out[1] == 1.0  # forward-fill from ts=0
    assert np.isnan(out[2])  # past last and future not allowed
    out2 = windows.asof_prev(ts, val, q, allow_future=True)
    assert out2[2] == 3.0  # past last allowed
