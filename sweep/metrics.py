"""
Prometheus Metrics for Sweep Engine

Complete observability for perp→spot sweeping operations.
"""

try:
    from prometheus_client import Counter, Gauge, Histogram
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


if _PROMETHEUS_AVAILABLE:
    # Counters
    SWEEP_SUCCESS_TOTAL = Counter(
        "xrpbot_sweep_success_total", 
        "Count of successful sweeps"
    )
    SWEEP_FAIL_TOTAL = Counter(
        "xrpbot_sweep_fail_total", 
        "Count of failed sweeps"
    )
    SWEEP_SKIPPED_TOTAL = Counter(
        "xrpbot_sweep_skipped_total", 
        "Count of skipped sweeps", 
        ["reason"]
    )

    # Gauges
    SWEEP_LAST_AMOUNT = Gauge(
        "xrpbot_sweep_last_amount", 
        "Last sweep amount (USDC)"
    )
    SWEEP_POSTBUF_BPS = Gauge(
        "xrpbot_sweep_post_buffer_bps", 
        "Post-sweep buffer (bps)"
    )
    SWEEP_EQUITY = Gauge(
        "xrpbot_sweep_equity_usdc", 
        "Equity (USDC)"
    )
    SWEEP_WITHDRAWABLE = Gauge(
        "xrpbot_sweep_withdrawable_usdc", 
        "Withdrawable (USDC)"
    )
    SWEEP_PENDING = Gauge(
        "xrpbot_sweep_pending_usdc", 
        "Accumulator pending (USDC or threshold)"
    )
    SWEEP_CD_REMAINING = Gauge(
        "xrpbot_sweep_cooldown_remaining_s", 
        "Cooldown remaining (s)"
    )
    
    # Histograms for detailed distribution analysis
    SWEEP_AMOUNT_HISTOGRAM = Histogram(
        "xrpbot_sweep_amount_histogram",
        "Distribution of sweep amounts (USDC)",
        buckets=[10, 20, 50, 100, 200, 500, 1000, float('inf')]
    )
    SWEEP_POST_BUFFER_HISTOGRAM = Histogram(
        "xrpbot_post_buffer_bps_histogram", 
        "Distribution of post-sweep buffer (bps)",
        buckets=[1500, 2000, 2500, 3000, 4000, 5000, float('inf')]
    )

else:
    # No-op stubs when prometheus_client is not available
    class _NoOpMetric:
        def labels(self, *args, **kwargs):
            return self
        
        def inc(self, *args, **kwargs):
            pass
        
        def set(self, *args, **kwargs):
            pass
        
        def observe(self, *args, **kwargs):
            pass

    SWEEP_SUCCESS_TOTAL = _NoOpMetric()
    SWEEP_FAIL_TOTAL = _NoOpMetric()
    SWEEP_SKIPPED_TOTAL = _NoOpMetric()
    SWEEP_LAST_AMOUNT = _NoOpMetric()
    SWEEP_POSTBUF_BPS = _NoOpMetric()
    SWEEP_EQUITY = _NoOpMetric()
    SWEEP_WITHDRAWABLE = _NoOpMetric()
    SWEEP_PENDING = _NoOpMetric()
    SWEEP_CD_REMAINING = _NoOpMetric()
    SWEEP_AMOUNT_HISTOGRAM = _NoOpMetric()
    SWEEP_POST_BUFFER_HISTOGRAM = _NoOpMetric()


__all__ = [
    "SWEEP_SUCCESS_TOTAL",
    "SWEEP_FAIL_TOTAL", 
    "SWEEP_SKIPPED_TOTAL",
    "SWEEP_LAST_AMOUNT",
    "SWEEP_POSTBUF_BPS",
    "SWEEP_EQUITY",
    "SWEEP_WITHDRAWABLE",
    "SWEEP_PENDING",
    "SWEEP_CD_REMAINING",
    "SWEEP_AMOUNT_HISTOGRAM",
    "SWEEP_POST_BUFFER_HISTOGRAM",
]
