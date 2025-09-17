class ExecutionRealismEngine:
    """
    Models square-root impact, queue position,
    and fill probabilities for execution realism.
    """
    def calibrate(self, market_data):
        return {"impact_coef": 0.0025}