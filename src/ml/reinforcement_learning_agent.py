"""
Reinforcement Learning Agent - Position Sizing with Bandit + Deep RL Hybrid
Implements RL agent for dynamic position sizing based on market conditions
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try to import deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class ActionType(Enum):
    SMALL_POSITION = "small_position"      # 0.1% - 0.5% of capital
    MEDIUM_POSITION = "medium_position"    # 0.5% - 1.0% of capital
    LARGE_POSITION = "large_position"      # 1.0% - 2.0% of capital
    NO_POSITION = "no_position"            # 0% of capital

class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

@dataclass
class State:
    market_regime: MarketRegime
    volatility: float
    momentum: float
    volume_ratio: float
    spread_bps: float
    funding_rate: float
    time_of_day: float  # 0-1 normalized hour
    day_of_week: float  # 0-1 normalized day
    recent_pnl: float
    current_drawdown: float
    position_size: float
    confidence_score: float

@dataclass
class Action:
    action_type: ActionType
    position_size: float
    confidence: float
    reasoning: str
    timestamp: str

@dataclass
class Reward:
    pnl: float
    risk_adjusted_return: float
    drawdown_penalty: float
    consistency_bonus: float
    total_reward: float
    timestamp: str

@dataclass
class Experience:
    state: State
    action: Action
    reward: Reward
    next_state: State
    done: bool
    timestamp: str

class MultiArmedBandit:
    """
    Multi-armed bandit for quick adaptation to changing conditions
    """
    
    def __init__(self, n_arms: int = 4):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.alpha = 0.1  # Learning rate
        self.epsilon = 0.1  # Exploration rate
        self.logger = logging.getLogger(__name__)
    
    def select_action(self) -> int:
        """Select action using epsilon-greedy strategy"""
        if np.random.random() < self.epsilon:
            # Explore
            return np.random.randint(self.n_arms)
        else:
            # Exploit
            return np.argmax(self.values)
    
    def update(self, arm: int, reward: float):
        """Update arm values"""
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        
        # Update using exponential moving average
        self.values[arm] = value + self.alpha * (reward - value)
    
    def get_arm_stats(self) -> Dict:
        """Get statistics for all arms"""
        return {
            "counts": self.counts.tolist(),
            "values": self.values.tolist(),
            "best_arm": int(np.argmax(self.values)),
            "exploration_rate": self.epsilon
        }

class DeepRLNetwork(nn.Module):
    """
    Deep neural network for RL agent
    """
    
    def __init__(self, state_size: int = 12, action_size: int = 4, hidden_size: int = 64):
        super(DeepRLNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return self.softmax(x)

class DeepRLAgent:
    """
    Deep RL agent using policy gradient method
    """
    
    def __init__(self, state_size: int = 12, action_size: int = 4, lr: float = 0.001):
        if not HAS_TORCH:
            self.logger = logging.getLogger(__name__)
            self.logger.warning("⚠️ PyTorch not available, using simplified RL agent")
            self.network = None
            return
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        
        self.network = DeepRLNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.logger = logging.getLogger(__name__)
        self.saved_log_probs = []
        self.rewards = []
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Select action using policy network"""
        if self.network is None:
            # Fallback to random action
            return np.random.randint(self.action_size), 0.5
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.network(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        self.saved_log_probs.append(dist.log_prob(action))
        return action.item(), action_probs[0][action].item()
    
    def update_policy(self):
        """Update policy using collected rewards"""
        if self.network is None or not self.saved_log_probs:
            return
        
        # Calculate discounted rewards
        R = 0
        policy_loss = []
        returns = []
        
        for r in self.rewards[::-1]:
            R = r + 0.99 * R  # Discount factor = 0.99
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize
        
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear saved data
        self.saved_log_probs = []
        self.rewards = []

class RLPositionSizingAgent:
    """
    Hybrid RL agent combining bandit and deep RL for position sizing
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.bandit = MultiArmedBandit(n_arms=4)
        self.deep_rl = DeepRLAgent()
        
        # Experience replay buffer
        self.experience_buffer: List[Experience] = []
        self.max_buffer_size = 10000
        
        # Action mapping
        self.action_mapping = {
            0: ActionType.NO_POSITION,
            1: ActionType.SMALL_POSITION,
            2: ActionType.MEDIUM_POSITION,
            3: ActionType.LARGE_POSITION
        }
        
        # Position size mapping
        self.position_sizes = {
            ActionType.NO_POSITION: 0.0,
            ActionType.SMALL_POSITION: 0.3,  # 0.3% of capital
            ActionType.MEDIUM_POSITION: 0.7,  # 0.7% of capital
            ActionType.LARGE_POSITION: 1.5   # 1.5% of capital
        }
        
        # Performance tracking
        self.performance_history: List[Reward] = []
        self.total_trades = 0
        self.winning_trades = 0
        
        # Create reports directory
        self.reports_dir = Path("reports/rl_agent")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_state(self, market_data: Dict, portfolio_data: Dict) -> State:
        """Extract state from market and portfolio data"""
        try:
            # Market regime detection
            volatility = market_data.get('volatility', 0.02)
            momentum = market_data.get('momentum', 0.0)
            volume_ratio = market_data.get('volume_ratio', 1.0)
            spread_bps = market_data.get('spread_bps', 5.0)
            funding_rate = market_data.get('funding_rate', 0.0)
            
            # Determine market regime
            if momentum > 0.02:
                market_regime = MarketRegime.TRENDING_UP
            elif momentum < -0.02:
                market_regime = MarketRegime.TRENDING_DOWN
            elif volatility > 0.05:
                market_regime = MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.01:
                market_regime = MarketRegime.LOW_VOLATILITY
            else:
                market_regime = MarketRegime.SIDEWAYS
            
            # Time features
            now = datetime.now()
            time_of_day = now.hour / 24.0
            day_of_week = now.weekday() / 7.0
            
            # Portfolio features
            recent_pnl = portfolio_data.get('recent_pnl', 0.0)
            current_drawdown = portfolio_data.get('current_drawdown', 0.0)
            position_size = portfolio_data.get('position_size', 0.0)
            confidence_score = portfolio_data.get('confidence_score', 0.5)
            
            return State(
                market_regime=market_regime,
                volatility=volatility,
                momentum=momentum,
                volume_ratio=volume_ratio,
                spread_bps=spread_bps,
                funding_rate=funding_rate,
                time_of_day=time_of_day,
                day_of_week=day_of_week,
                recent_pnl=recent_pnl,
                current_drawdown=current_drawdown,
                position_size=position_size,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            self.logger.error(f"❌ State extraction error: {e}")
            # Return default state
            return State(
                market_regime=MarketRegime.SIDEWAYS,
                volatility=0.02,
                momentum=0.0,
                volume_ratio=1.0,
                spread_bps=5.0,
                funding_rate=0.0,
                time_of_day=0.5,
                day_of_week=0.5,
                recent_pnl=0.0,
                current_drawdown=0.0,
                position_size=0.0,
                confidence_score=0.5
            )
    
    def state_to_vector(self, state: State) -> np.ndarray:
        """Convert state to numerical vector"""
        regime_encoding = {
            MarketRegime.TRENDING_UP: [1, 0, 0, 0, 0],
            MarketRegime.TRENDING_DOWN: [0, 1, 0, 0, 0],
            MarketRegime.SIDEWAYS: [0, 0, 1, 0, 0],
            MarketRegime.HIGH_VOLATILITY: [0, 0, 0, 1, 0],
            MarketRegime.LOW_VOLATILITY: [0, 0, 0, 0, 1]
        }
        
        regime_vec = regime_encoding[state.market_regime]
        
        # Ensure we have exactly 12 features to match the network input size
        state_vector = regime_vec + [
            state.volatility,
            state.momentum,
            state.volume_ratio,
            state.spread_bps / 100.0,  # Normalize
            state.funding_rate,
            state.time_of_day,
            state.day_of_week,
            state.recent_pnl / 1000.0,  # Normalize
            state.current_drawdown,
            state.position_size,
            state.confidence_score
        ]
        
        # Pad or truncate to exactly 12 features
        if len(state_vector) > 12:
            state_vector = state_vector[:12]
        elif len(state_vector) < 12:
            state_vector.extend([0.0] * (12 - len(state_vector)))
        
        return np.array(state_vector, dtype=np.float32)
    
    def select_action(self, state: State) -> Action:
        """Select action using hybrid approach"""
        try:
            # Use bandit for quick adaptation
            bandit_action = self.bandit.select_action()
            bandit_confidence = 0.7
            
            # Use deep RL for complex decisions
            if self.deep_rl.network is not None:
                state_vector = self.state_to_vector(state)
                rl_action, rl_confidence = self.deep_rl.select_action(state_vector)
                
                # Combine decisions (weighted average)
                if rl_confidence > 0.8:  # High confidence in RL
                    final_action = rl_action
                    confidence = rl_confidence
                    reasoning = "Deep RL high confidence"
                else:
                    final_action = bandit_action
                    confidence = bandit_confidence
                    reasoning = "Bandit fallback"
            else:
                final_action = bandit_action
                confidence = bandit_confidence
                reasoning = "Bandit only (no deep RL)"
            
            action_type = self.action_mapping[final_action]
            position_size = self.position_sizes[action_type]
            
            return Action(
                action_type=action_type,
                position_size=position_size,
                confidence=confidence,
                reasoning=reasoning,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Action selection error: {e}")
            # Return safe action
            return Action(
                action_type=ActionType.SMALL_POSITION,
                position_size=0.3,
                confidence=0.1,
                reasoning="Error fallback",
                timestamp=datetime.now().isoformat()
            )
    
    def calculate_reward(self, action: Action, trade_result: Dict) -> Reward:
        """Calculate reward based on trade result"""
        try:
            pnl = trade_result.get('pnl', 0.0)
            trade_duration = trade_result.get('duration', 1.0)
            max_drawdown = trade_result.get('max_drawdown', 0.0)
            
            # Base reward from PnL
            pnl_reward = pnl / 100.0  # Normalize to reasonable scale
            
            # Risk-adjusted return
            if trade_duration > 0:
                risk_adjusted_return = pnl / (max_drawdown + 0.01) / trade_duration
            else:
                risk_adjusted_return = 0.0
            
            # Drawdown penalty
            drawdown_penalty = -max_drawdown * 10.0
            
            # Consistency bonus (reward for consistent performance)
            consistency_bonus = 0.0
            if len(self.performance_history) > 10:
                recent_rewards = [r.pnl for r in self.performance_history[-10:]]
                if len(recent_rewards) > 0:
                    consistency = 1.0 - np.std(recent_rewards) / (np.mean(np.abs(recent_rewards)) + 0.01)
                    consistency_bonus = consistency * 0.1
            
            # Total reward
            total_reward = pnl_reward + risk_adjusted_return * 0.1 + drawdown_penalty + consistency_bonus
            
            return Reward(
                pnl=pnl,
                risk_adjusted_return=risk_adjusted_return,
                drawdown_penalty=drawdown_penalty,
                consistency_bonus=consistency_bonus,
                total_reward=total_reward,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Reward calculation error: {e}")
            return Reward(
                pnl=0.0,
                risk_adjusted_return=0.0,
                drawdown_penalty=0.0,
                consistency_bonus=0.0,
                total_reward=0.0,
                timestamp=datetime.now().isoformat()
            )
    
    def update_agent(self, experience: Experience):
        """Update agent with new experience"""
        try:
            # Add to experience buffer
            self.experience_buffer.append(experience)
            if len(self.experience_buffer) > self.max_buffer_size:
                self.experience_buffer = self.experience_buffer[-self.max_buffer_size:]
            
            # Update bandit
            action_idx = list(self.action_mapping.keys())[list(self.action_mapping.values()).index(experience.action.action_type)]
            self.bandit.update(action_idx, experience.reward.total_reward)
            
            # Update deep RL
            if self.deep_rl.network is not None:
                self.deep_rl.rewards.append(experience.reward.total_reward)
                
                # Update policy every 10 experiences
                if len(self.deep_rl.rewards) >= 10:
                    self.deep_rl.update_policy()
            
            # Track performance
            self.performance_history.append(experience.reward)
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
            # Update trade statistics
            self.total_trades += 1
            if experience.reward.pnl > 0:
                self.winning_trades += 1
            
        except Exception as e:
            self.logger.error(f"❌ Agent update error: {e}")
    
    def get_agent_stats(self) -> Dict:
        """Get agent performance statistics"""
        if not self.performance_history:
            return {"message": "No performance data available"}
        
        recent_rewards = [r.total_reward for r in self.performance_history[-100:]]
        recent_pnls = [r.pnl for r in self.performance_history[-100:]]
        
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": win_rate,
            "average_reward": np.mean(recent_rewards),
            "average_pnl": np.mean(recent_pnls),
            "total_pnl": sum(recent_pnls),
            "bandit_stats": self.bandit.get_arm_stats(),
            "experience_buffer_size": len(self.experience_buffer),
            "performance_history_size": len(self.performance_history)
        }
    
    async def save_agent_state(self):
        """Save agent state to file"""
        try:
            state_data = {
                "bandit_stats": self.bandit.get_arm_stats(),
                "performance_history": [asdict(r) for r in self.performance_history[-100:]],
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "timestamp": datetime.now().isoformat()
            }
            
            state_file = self.reports_dir / "rl_agent_state.json"
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            self.logger.info(f"💾 Agent state saved: {state_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save agent state: {e}")

# Demo function
async def demo_rl_agent():
    """Demo the RL position sizing agent"""
    print("🤖 RL Position Sizing Agent Demo")
    print("=" * 50)
    
    # Create RL agent
    agent = RLPositionSizingAgent()
    
    # Simulate trading episodes
    print("🎮 Simulating trading episodes...")
    
    for episode in range(20):
        # Generate market data
        market_data = {
            'volatility': np.random.uniform(0.01, 0.05),
            'momentum': np.random.uniform(-0.03, 0.03),
            'volume_ratio': np.random.uniform(0.5, 2.0),
            'spread_bps': np.random.uniform(2.0, 10.0),
            'funding_rate': np.random.uniform(-0.01, 0.01)
        }
        
        # Generate portfolio data
        portfolio_data = {
            'recent_pnl': np.random.uniform(-100, 100),
            'current_drawdown': np.random.uniform(0, 0.05),
            'position_size': np.random.uniform(0, 0.02),
            'confidence_score': np.random.uniform(0.3, 0.9)
        }
        
        # Extract state
        state = agent.extract_state(market_data, portfolio_data)
        
        # Select action
        action = agent.select_action(state)
        
        # Simulate trade result
        trade_result = {
            'pnl': np.random.uniform(-50, 150),
            'duration': np.random.uniform(0.5, 5.0),
            'max_drawdown': np.random.uniform(0, 0.02)
        }
        
        # Calculate reward
        reward = agent.calculate_reward(action, trade_result)
        
        # Create experience
        next_state = agent.extract_state(market_data, portfolio_data)  # Simplified
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=False,
            timestamp=datetime.now().isoformat()
        )
        
        # Update agent
        agent.update_agent(experience)
        
        if episode % 5 == 0:
            print(f"Episode {episode}: {action.action_type.value} (confidence: {action.confidence:.3f}, reward: {reward.total_reward:.3f})")
    
    # Get agent statistics
    stats = agent.get_agent_stats()
    print(f"\n📊 Agent Statistics:")
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Win Rate: {stats['win_rate']:.1%}")
    print(f"Average Reward: {stats['average_reward']:.3f}")
    print(f"Average PnL: {stats['average_pnl']:.2f}")
    print(f"Total PnL: {stats['total_pnl']:.2f}")
    
    print(f"\n🎰 Bandit Statistics:")
    bandit_stats = stats['bandit_stats']
    print(f"Best Arm: {bandit_stats['best_arm']}")
    print(f"Arm Values: {[f'{v:.3f}' for v in bandit_stats['values']]}")
    print(f"Arm Counts: {bandit_stats['counts']}")
    
    # Save agent state
    await agent.save_agent_state()
    
    print("\n✅ RL Agent Demo Complete")

if __name__ == "__main__":
    asyncio.run(demo_rl_agent())
