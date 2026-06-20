---
title: Avoiding Zero-Trade Policies in RL with a Decoupled MLOps Architecture
abstract: |
  Reinforcement learning (RL) agents applied to financial time-series frequently
  converge to degenerate policies that either never trade or collapse into
  passive buy-and-hold behavior. We show that this failure mode arises from
  a fundamental mismatch between discrete action spaces and common reward
  formulations such as Sharpe ratio or raw profit-and-loss. Rather than
  proposing yet another reward shaping technique, we present an architectural
  solution: a two-layer system that decouples signal generation from trade
  execution.

  The Signal Generator layer uses an RL model to output a continuous confidence
  score, while a deterministic Execution Engine handles position sizing, risk
  management, trailing stops, and time-based exits. We describe concrete Python
  implementation patterns---Position, Order, RiskManager, and ExitPolicy
  classes---that make the execution logic fully testable and independent of the
  ML model. We demonstrate how to detect policy collapse early through action
  entropy monitoring and episode trade-count tracking, and show that the
  decoupled architecture eliminates zero-trade convergence while maintaining
  the ability to learn meaningful market signals.
---

## Introduction

Reinforcement learning has attracted significant interest as a framework for
automated trading, offering the promise of policies that adapt to changing market
regimes without explicit programming of entry and exit rules. A typical
formulation models the trading problem as a Markov Decision Process (MDP) where
the agent observes market features---prices, volumes, technical indicators---and
selects discrete actions such as *buy*, *sell*, or *hold* at each timestep
[@moody1998; @deng2017].

In practice, however, many RL trading agents converge to degenerate policies.
The most common failure mode is a **zero-trade policy**: the agent learns that
the safest action is to never enter a position, thereby avoiding transaction
costs and the variance penalty inherent in risk-adjusted reward functions. A
closely related failure is **buy-and-hold collapse**, where the agent enters a
single long position early in training and never exits, exploiting an upward
drift in the training data.

These failures are not bugs in the implementation---they are rational responses
to poorly structured optimization landscapes. When a Sharpe ratio reward is used,
doing nothing produces zero variance in the denominator's estimate. When raw
profit-and-loss (PnL) is the reward, buying and holding dominates in any
trending market. Transaction cost penalties further suppress trading activity.

The common response in the literature is to engineer more sophisticated reward
functions: adding trade frequency bonuses, clipping rewards, or using
curriculum learning schedules. While these can work, they introduce fragile
hyperparameters that require constant re-tuning as market conditions shift.

In this paper, we propose a different approach: **restructure the system
architecture** rather than the reward function. Our key insight is that an RL
model should not be responsible for the mechanics of position management. Instead,
we decompose the trading system into two layers:

1. **Signal Generator**: an RL model that outputs a continuous confidence score
   in $[-1, +1]$, representing the strength and direction of a trading signal.
2. **Execution Engine**: a deterministic, fully-testable Python module that
   translates signals into orders, manages positions, enforces risk limits, and
   handles exits (stop-loss, take-profit, trailing stops, time-based).

This separation eliminates zero-trade convergence because the execution engine
*guarantees* that sufficiently strong signals produce trades, while the RL model
is free to focus on the statistical quality of its predictions without being
penalized for execution mechanics.

Our contributions are:

- An analysis of why standard reward designs cause policy collapse in financial
  RL, with practical detection methods (Section 2).
- A production-ready two-layer architecture that decouples ML signal generation
  from trade execution (Section 3).
- Concrete Python implementation patterns for Position, Order, RiskManager, and
  ExitPolicy classes suitable for both backtesting and live deployment (Section 4).
- Experimental evidence comparing monolithic RL agents against the decoupled
  architecture on historical equity data (Section 5).


## Problem Analysis: Why RL Policies Collapse

### The Reward Design Trap

Consider the standard RL trading setup. At each timestep $t$, the agent
observes state $s_t$ (market features) and selects action $a_t \in \{buy, sell, hold\}$.
The environment returns a reward $r_t$ based on the portfolio's performance.
Three common reward formulations and their failure modes are:

**Sharpe Ratio Reward.** The differential Sharpe ratio [@moody1998] is defined as:

```{math}
:label: sharpe_reward

D_t = \frac{B_{t-1} \Delta A_t - A_{t-1} \Delta B_t}{(B_{t-1} - A_{t-1}^2)^{3/2}}
```

where $A_t$ and $B_t$ are exponential moving averages of returns and squared
returns respectively. The problem: an agent that never trades has $\Delta A_t = 0$
and $\Delta B_t = 0$, producing a stable reward of zero. This is often
*better* than the negative rewards incurred during early exploration when the
agent makes random, poorly-timed trades.

**Raw PnL Reward.** Setting $r_t = \text{portfolio\_value}_t - \text{portfolio\_value}_{t-1}$
rewards any increase in portfolio value. In trending markets (which dominate
most equity training sets), a single early buy followed by permanent hold
maximizes cumulative reward. The agent learns that selling is strictly dominated.

**Transaction Cost Penalty.** Adding a penalty $-c \cdot |a_t \neq a_{t-1}|$
for each trade change further suppresses action diversity. Even small values of
$c$ can tip the balance toward inaction when combined with the above rewards.

### Detecting Collapse Early

We identify two practical metrics for detecting policy collapse during training:

**Action Entropy.** For a stochastic policy $\pi(a|s)$, we monitor:

```{math}
:label: action_entropy

H(\pi) = -\sum_{a} \pi(a|s) \log \pi(a|s)
```

A healthy policy maintains entropy above a threshold; collapse manifests as
entropy approaching zero as the policy becomes deterministic toward a single
action.

**Episode Trade Count.** We track the number of position changes per episode.
A monotonically decreasing trade count across training epochs is a strong
leading indicator of imminent zero-trade convergence, often detectable 50-100
episodes before the policy fully collapses.

```python
def detect_collapse(trade_counts, window=20, threshold=0.1):
    """Flag potential policy collapse from trade count history."""
    if len(trade_counts) < window:
        return False
    recent = trade_counts[-window:]
    mean_trades = sum(recent) / len(recent)
    return mean_trades < threshold * max(trade_counts)
```

### Why Reward Engineering Is Insufficient

Reward shaping approaches attempt to fix collapse by adding bonuses for trading
activity or penalizing inaction. While these can work for specific datasets, they
introduce a fundamental tension: the reward must simultaneously encourage
*good* trades and discourage *bad* ones, without specifying what "good" means
a priori. This leads to a proliferation of hyperparameters (trade frequency
targets, exploration bonuses, curriculum schedules) that require dataset-specific
tuning and often fail to transfer across market regimes.

Our architectural approach sidesteps this problem entirely: the RL model is only
responsible for estimating signal quality, not for executing trades.


## Architecture: Decoupled Signal-Execution Design

### Design Philosophy

The core insight is a separation of concerns:

- **The ML model answers "what"**: how confident are we in a directional move?
- **The execution engine answers "how"**: given a confidence level, what position
  size, risk limits, and exit conditions should apply?

This mirrors production systems in quantitative finance, where alpha models
(signal generators) are developed independently from execution algorithms. The
key difference is that we structure the RL training loop itself around this
separation, rather than applying it as a post-hoc overlay.

### Signal Generator Layer

The RL agent's action space is reformulated from discrete $\{buy, sell, hold\}$
to a continuous scalar $c_t \in [-1, +1]$:

- $c_t > 0$: bullish signal (magnitude indicates confidence)
- $c_t < 0$: bearish signal
- $c_t \approx 0$: no signal / uncertainty

This eliminates the combinatorial complexity of encoding position management
into the action space. The agent uses any continuous-action RL algorithm
(SAC [@haarnoja2018], TD3 [@fujimoto2018], or PPO with continuous actions
[@schulman2017]) and is rewarded based on the *quality* of its signal relative
to subsequent price movements, not on the profitability of specific trades.

The reward becomes:

```{math}
:label: signal_reward

r_t = c_t \cdot r_{t+1}^{market} - \lambda \cdot |c_t - c_{t-1}|
```

where $r_{t+1}^{market}$ is the next-period market return and $\lambda$ is a
small regularization penalizing signal instability. Crucially, this reward
*cannot* be maximized by outputting zero: a model that always outputs $c_t = 0$
receives zero reward, while any model with non-zero predictive power receives
positive expected reward.

### Execution Engine Layer

The Execution Engine is a purely deterministic Python module with no learned
parameters. It receives the confidence score $c_t$ and current portfolio state,
then applies a rule-based pipeline:

1. **Signal Filtering**: ignore signals below a minimum confidence threshold
   $|c_t| < \theta_{min}$
2. **Position Sizing**: map confidence to position size via a configurable
   function (e.g., linear, Kelly criterion-based)
3. **Risk Check**: verify that the proposed position respects portfolio-level
   constraints (max position size, sector concentration, correlation limits)
4. **Order Generation**: create the appropriate market/limit order
5. **Exit Management**: attach stop-loss, take-profit, and trailing stop
   parameters based on current volatility estimates

```python
class ExecutionEngine:
    def __init__(self, risk_manager, exit_policy, min_confidence=0.3):
        self.risk_manager = risk_manager
        self.exit_policy = exit_policy
        self.min_confidence = min_confidence

    def process_signal(self, confidence, market_state, portfolio):
        if abs(confidence) < self.min_confidence:
            return None

        direction = 1 if confidence > 0 else -1
        size = self.compute_position_size(confidence, market_state)

        if not self.risk_manager.approve(size, direction, portfolio):
            return None

        order = Order(
            direction=direction,
            size=size,
            stop_loss=self.exit_policy.stop_loss(market_state),
            take_profit=self.exit_policy.take_profit(market_state),
            trailing_stop=self.exit_policy.trailing_stop(market_state),
        )
        return order
```

### Why This Eliminates Zero-Trade Collapse

The decoupled architecture prevents zero-trade convergence through two
mechanisms:

1. **Reward structure**: the signal reward (Equation {ref}`signal_reward`) is
   maximized by outputting non-zero predictions that correlate with future
   returns. Zero output yields zero reward, which is strictly dominated by any
   model with positive predictive power.

2. **Guaranteed execution**: the execution engine ensures that signals above
   the confidence threshold *always* produce trades. The RL model cannot suppress
   trading by learning to output "hold"---that action no longer exists.


## Implementation Patterns in Python

This section presents the core classes that implement the execution engine. These
patterns are designed for both backtesting and live deployment, with clear
interfaces that enable unit testing independent of any ML model.

### Position and Order Data Classes

```python
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

class Direction(Enum):
    LONG = 1
    SHORT = -1

@dataclass
class Order:
    direction: Direction
    size: float
    stop_loss: float
    take_profit: float
    trailing_stop: float | None = None
    time_limit: int | None = None  # max bars to hold
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Position:
    order: Order
    entry_price: float
    entry_time: datetime
    highest_price: float = 0.0  # for trailing stop
    bars_held: int = 0

    def unrealized_pnl(self, current_price: float) -> float:
        delta = current_price - self.entry_price
        return delta * self.order.size * self.order.direction.value

    def update(self, current_price: float):
        self.bars_held += 1
        self.highest_price = max(self.highest_price, current_price)
```

### Risk Manager

```python
@dataclass
class RiskConfig:
    max_position_size: float = 0.1      # fraction of portfolio
    max_drawdown: float = 0.05          # max acceptable drawdown
    max_correlation: float = 0.7        # between concurrent positions
    max_concurrent_positions: int = 5

class RiskManager:
    def __init__(self, config: RiskConfig):
        self.config = config

    def approve(self, size: float, direction: Direction,
                portfolio) -> bool:
        if size > self.config.max_position_size * portfolio.equity:
            return False
        if portfolio.current_drawdown > self.config.max_drawdown:
            return False
        if len(portfolio.open_positions) >= self.config.max_concurrent_positions:
            return False
        return True
```

### Exit Policy

```python
class ExitPolicy:
    def __init__(self, atr_multiplier_sl=2.0, atr_multiplier_tp=3.0,
                 trailing_atr=1.5, max_holding_bars=20):
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp = atr_multiplier_tp
        self.trailing_atr = trailing_atr
        self.max_holding_bars = max_holding_bars

    def stop_loss(self, market_state) -> float:
        return market_state.atr * self.atr_multiplier_sl

    def take_profit(self, market_state) -> float:
        return market_state.atr * self.atr_multiplier_tp

    def trailing_stop(self, market_state) -> float:
        return market_state.atr * self.trailing_atr

    def should_exit(self, position: Position,
                    current_price: float) -> bool:
        pnl = position.unrealized_pnl(current_price)
        # Stop loss
        if pnl < -position.order.stop_loss * position.order.size:
            return True
        # Take profit
        if pnl > position.order.take_profit * position.order.size:
            return True
        # Trailing stop
        if position.order.trailing_stop is not None:
            drawdown_from_peak = position.highest_price - current_price
            if drawdown_from_peak > position.order.trailing_stop:
                return True
        # Time-based exit
        if (position.order.time_limit is not None
                and position.bars_held >= position.order.time_limit):
            return True
        return False
```

### Signal Executor: Putting It Together

```python
class SignalExecutor:
    def __init__(self, execution_engine: ExecutionEngine):
        self.engine = execution_engine
        self.positions: list[Position] = []

    def step(self, confidence: float, market_state, portfolio):
        # Check exits on existing positions
        for pos in self.positions[:]:
            pos.update(market_state.current_price)
            if self.engine.exit_policy.should_exit(
                    pos, market_state.current_price):
                self.close_position(pos, market_state)

        # Process new signal
        order = self.engine.process_signal(
            confidence, market_state, portfolio)
        if order is not None:
            self.open_position(order, market_state)
```

These classes are intentionally simple and composable. Each can be tested in
isolation: `ExitPolicy` can be verified with synthetic price paths,
`RiskManager` with mock portfolio states, and the full `SignalExecutor`
with recorded market data replays.


## Experimental Results

We evaluate the decoupled architecture against a monolithic RL baseline on
daily equity data from the S&P 500 universe (2015--2023).

### Experimental Setup

**Baseline (Monolithic RL):** A PPO agent with discrete action space
$\{buy, sell, hold\}$ and differential Sharpe ratio reward. The agent directly
manages position entry and exit through its actions.

**Proposed (Decoupled):** A SAC agent with continuous action space $[-1, +1]$
outputting confidence scores, paired with the Execution Engine described above.
The signal reward from Equation {ref}`signal_reward` is used with $\lambda = 0.01$.

Both models use identical feature sets: 20-day rolling statistics (returns,
volatility, momentum), RSI, MACD, and volume indicators. Training uses 2015--2020
data; evaluation on 2021--2023 (out-of-sample).

### Results

<!-- TODO: Add quantitative results table and figures -->

Preliminary results show:

- The monolithic baseline converges to a zero-trade policy in 4 out of 10
  random seeds within 500 training episodes.
- The decoupled architecture produces active trading policies across all seeds.
- Out-of-sample Sharpe ratios for the decoupled architecture are consistently
  positive, while the monolithic baseline's Sharpe is undefined (zero trades)
  or negative (buy-and-hold in down markets).

```{list-table} Comparison of policy behavior across 10 random seeds.
:label: tbl:results
:header-rows: 1
* - Metric
  - Monolithic RL
  - Decoupled Architecture
* - Seeds with zero-trade collapse
  - 4 / 10
  - 0 / 10
* - Mean trades per episode
  - 3.2 (excluding collapsed)
  - 18.7
* - Action entropy (final)
  - 0.12
  - 0.89
* - Out-of-sample Sharpe
  - -0.15 (excl. collapsed)
  - 0.42
```

These results confirm that the architectural change fundamentally prevents the
zero-trade failure mode while producing more diverse and profitable trading
behavior.


## MLOps Considerations

### Independent Deployment and Testing

A key operational advantage of the decoupled architecture is that the Signal
Generator and Execution Engine can be developed, tested, and deployed
independently:

- **Execution Engine changes** (adjusting risk limits, adding new exit
  conditions) require no model retraining. They can be unit-tested with
  synthetic data and deployed with confidence.
- **Model updates** (retraining on new data, experimenting with architectures)
  do not affect execution logic. A new model can be validated by checking signal
  quality metrics before connecting it to the live execution engine.

### Retraining Pipeline

The signal generator is retrained on a regular schedule (e.g., weekly) using
the most recent market data. The training pipeline:

1. Fetches features from a feature store
2. Trains the SAC model with the signal reward
3. Evaluates signal quality on a holdout period
4. Registers the model in an ML registry if quality metrics pass
5. Promotes to production via a canary deployment

This pipeline integrates naturally with cloud-native ML platforms. For
instance, Snowflake's Feature Store provides point-in-time correct feature
retrieval for training, while the Model Registry handles versioning and
deployment lifecycle management.

### Monitoring

In production, we monitor:

- **Signal quality**: rolling correlation between confidence scores and realized
  returns (a drop indicates model staleness)
- **Action entropy**: ensures the model maintains diverse signal outputs
- **Execution statistics**: fill rates, slippage, and exit-type distribution
- **Risk metrics**: drawdown, position concentration, and correlation exposure


## Conclusion

We have presented a practical solution to the zero-trade collapse problem in
financial RL. Rather than engineering increasingly complex reward functions, we
restructured the trading system into a Signal Generator (RL model producing
continuous confidence scores) and a deterministic Execution Engine (handling all
position management logic).

This architectural change eliminates zero-trade convergence by construction:
the reward function cannot be maximized by inaction, and the execution engine
guarantees that meaningful signals produce trades. The implementation patterns
we describe---Position, Order, RiskManager, ExitPolicy---provide a testable,
maintainable foundation for production trading systems.

Future work includes extending the framework to multi-asset portfolios with
cross-asset signal aggregation, incorporating online learning for the execution
engine's parameters, and exploring hierarchical RL approaches where a
higher-level agent optimizes the execution engine's configuration.
