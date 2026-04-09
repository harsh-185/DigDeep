from agent.memory.cost_tracker import CostTracker


def test_initial_state():
    tracker = CostTracker()
    assert tracker.total_cost == 0.0
    assert tracker.total_input_tokens == 0
    assert tracker.total_output_tokens == 0
    assert len(tracker.call_log) == 0


def test_record_call():
    tracker = CostTracker()
    result = tracker.record_call("input text here", "output text", "test_purpose")
    assert result["cost"] > 0
    assert result["cumulative"] > 0
    assert len(tracker.call_log) == 1
    assert tracker.call_log[0]["purpose"] == "test_purpose"


def test_multiple_calls_accumulate():
    tracker = CostTracker()
    tracker.record_call("input 1", "output 1", "call_1")
    tracker.record_call("input 2", "output 2", "call_2")
    assert len(tracker.call_log) == 2
    assert tracker.total_cost == tracker.call_log[0]["cost"] + tracker.call_log[1]["cost"]


def test_can_afford_within_budget():
    tracker = CostTracker()
    assert tracker.can_afford(100) is True


def test_can_afford_over_budget():
    tracker = CostTracker()
    # Simulate spending almost all budget
    tracker.total_cost = 0.099
    assert tracker.can_afford(10000) is False


def test_get_report():
    tracker = CostTracker()
    tracker.record_call("hello world", "response", "test")
    report = tracker.get_report()
    assert "total_cost" in report
    assert "budget_remaining" in report
    assert "budget_used_pct" in report
    assert "call_log" in report
    assert report["budget_remaining"] > 0
