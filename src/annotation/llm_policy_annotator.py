class LLMPPolicyAnnotator:
    """
    LLM Policy Annotator

    Reviews RL agent actions before execution.

    Possible outcomes:
    - approve
    - reduce
    - reject
    """

    def __init__(
        self,
        max_position_size=500,
        drawdown_limit=0.20,
        risk_reduction_factor=0.5
    ):

        self.max_position_size = max_position_size
        self.drawdown_limit = drawdown_limit
        self.risk_reduction_factor = risk_reduction_factor


    def review_action(self, action, portfolio_value, drawdown):

        decision = "approve"

        adjusted_action = action.copy()

        # Rule 1 — excessive position size
        if max(abs(action)) > self.max_position_size:

            decision = "reduce"

            adjusted_action = action * self.risk_reduction_factor

        # Rule 2 — portfolio drawdown protection
        if drawdown > self.drawdown_limit:

            decision = "reject"

            adjusted_action = action * 0

        return decision, adjusted_action
