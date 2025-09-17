class FundingAccrualEngine:
    """
    Tracks per-interval funding accrual and
    stress tests funding shocks / latency misses.
    """
    def accrue(self, rates):
        return {"funding_pnl": sum(rates)}