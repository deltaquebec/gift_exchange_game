#!/usr/bin/env python3
"""

GIFT EXCHANGE SIMULATION

Author: Daniel Quigley
Version: 3.4

Simulates gift exchange games. We have some configurable options in the CLI:

TODO: break this into multiple scripts; ~1400 lines is too much....


VALUATION MODELS (--valuation-model):

    independent     : 	each player's preferences are i.i.d. random draws.
    					no correlation across players
					    
    correlated (default)     :	players share a common "objective quality" component plus
                      				idiosyncratic noise. Some gifts are universally better. 
    
    neg_correlated  : two "camps" with opposing preferences. What camp A loves,
                      			camp B dislikes
    
    all             : run experiment across ALL valuation models (for --ablation,
                      		--isolation, --additive modes).


BEHAVIORAL FEATURES (--features):

    partial_info    : players only know true values of OPENED gifts. Wrapped
                      		gifts have noisy "appearance signals" (size, wrapping
                      		quality). Players form Bayesian(-ish) estimates. Models
                      		real uncertainty about unopened gifts.
    
    social_costs    : stealing incurs utility penalties:
                      		- base cost per steal (awkwardness);
                      		- repeat-offense multiplier (stealing from same person);
                      		- reputation decay (serial stealer stigma).
                      		Models relationship/social friction that regulates
                      		aggressive play in real games.
    
    adaptive        : players dynamically adjust strategy based on:
                      		- game phase (more aggressive late-game);
                      		- current satisfaction (happy players steal less);
                      		- frustration level (victims become more aggressive).
                      		Models emotional/reactive human behavior.
    
    biased_selection: when opening a gift, players do not choose uniformly
                      			random. They're drawn to promising-looking gifts
                      			(softmax over appearance signals). Models peeking,
                      			guessing by wrapping, etc.


EXPERIMENT MODES:

    --isolation     : compare BASE (no features) vs. each single feature vs.
                      		FULL (all features). 6 experiments. Identifies marginal
                      		effect of each feature in isolation.
    
    --ablation      : run all 2^4 = 16 combinations of features. Full factorial
                      		design. Identifies interaction effects between features.
                      		With --valuation-model all, runs across all 3 models (48 total).
    
    --additive      : add features incrementally: BASE → +PI → +PI+SC → 
                      		+PI+SC+AD → FULL. Shows cumulative effect of layering.
    
    --features X Y  : run single experiment with specified features enabled.
                      		Use 'all' for full decoration, 'none' for base game.


OUTPUT:

    - Console: summary statistics, rankings, marginal effects
    - .txt file: all reports saved to {output_dir}/report.txt
    - .png files: individual plots saved to {output_dir}/:
        - steals_per_game.png
        - chain_length.png
        - seat_advantage.png
        - strategy_performance.png
        - seat_heatmap.png (if applicable)
        - strategy_ranking.png


USAGE EXAMPLES:

    # run feature isolation study
    python white_elephant.py --isolation
    
    # run base game only
    python white_elephant.py --features none
    
    # run fully decorated game
    python white_elephant.py --features all
    
    # full ablation across all valuation models
    python white_elephant.py --ablation --valuation-model all
    
    # ablation for specific models
    python white_elephant.py --ablation --valuation-model correlated independent
    
    # custom output directory
    python white_elephant.py --isolation --output-dir ./results


"""

import argparse
import os
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from itertools import combinations
from typing import Optional, Dict, Set, List, Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable



# CONFIGURATION


FEATURES = ['partial_info', 'social_costs', 'adaptive', 'biased_selection']

FEATURE_DESCRIPTIONS = {
    'partial_info': 'partial Information (uncertainty about wrapped gifts)',
    'social_costs': 'social costs (reputation/relationship penalties)',
    'adaptive': 'adaptive satrategy (dynamic behavior adjustment)',
    'biased_selection': 'biased selection (appearance-based gift choice)',
}

VALUATION_MODELS = ['independent', 'correlated', 'neg_correlated']


@dataclass
class GameConfig:
    """All tunable parameters for the simulation"""
    # Core settings
    num_players: int = 29
    num_games: int = 10_000
    seed: int = 42
    
    # valuation model
    valuation_model: str = 'correlated'
    correlation_strength: float = 0.7
    
    # feature toggles
    enable_partial_information: bool = False
    enable_social_costs: bool = False
    enable_adaptive_strategy: bool = False
    enable_biased_selection: bool = False
    
    # partial information parameters (Bayesian belief formation)
    signal_noise: float = 0.3          # σ_a: std dev of appearance signal noise
    prior_mean: float = 0.5            # μ_0: prior mean of gift quality
    prior_variance: float = 0.25       # σ_0²: prior variance (0.25 = std dev 0.5)
    risk_aversion: float = 0.5         # ρ: CARA risk aversion coefficient
    
    # social cost parameters
    steal_base_cost: float = 0.05         # c_0: norm violation disutility
    repeat_steal_multiplier: float = 2.0  # α: relationship damage multiplier
    reputation_decay: float = 0.1         # β: marginal reputation cost per steal
    
    # adaptive strategy parameters (logit choice model)
    frustration_gain: float = 0.15         # γ: frustration increment per theft
    frustration_decay: float = 0.05        # γ': frustration decay per round
    lambda_phase: float = 0.2              # λ_1: phase effect on steal probability
    lambda_frustration: float = 0.5        # λ_2: frustration effect on steal probability  
    lambda_satisfaction: float = 0.3       # λ_3: satisfaction dampening effect
    
    # stealing limit parameters
    # TODO set this as controlable from the start....
    max_steals_per_round: int = 1          # m,ax times steal in one round (0 = unlimited)
    max_steals_total: int = 0              # max times steal across entire game (0 = unlimited)


def config_from_features(features: List[str], num_players: int = 29, num_games: int = 5000, seed: int = 42, valuation_model: str = 'correlated', max_steals_per_round: int = 1, max_steals_total: int = 0) -> GameConfig:
    """Create GameConfig with specified features enabled"""
    return GameConfig(
        num_players=num_players,
        num_games=num_games,
        seed=seed,
        valuation_model=valuation_model,
        enable_partial_information='partial_info' in features,
        enable_social_costs='social_costs' in features,
        enable_adaptive_strategy='adaptive' in features,
        enable_biased_selection='biased_selection' in features,
        max_steals_per_round=max_steals_per_round,
        max_steals_total=max_steals_total,
    )


def config_label(config: GameConfig) -> str:
    """Generate readable label for a configuration"""
    flags = []
    if config.enable_partial_information:
        flags.append('PI')
    if config.enable_social_costs:
        flags.append('SC')
    if config.enable_adaptive_strategy:
        flags.append('AD')
    if config.enable_biased_selection:
        flags.append('BS')
    
    if not flags:
        return 'BASE'
    elif len(flags) == 4:
        return 'FULL'
    else:
        return '+'.join(flags)



# CONTROL OUTPUT HERE (TODO: check for multiple locations?)


class OutputManager:
    """Manages console output, file output, and plot generation"""
    
    def __init__(self, output_dir: str = './output'):
        self.output_dir = output_dir
        self.report_buffer = StringIO()
        os.makedirs(output_dir, exist_ok=True)
    
    def print(self, *args, **kwargs):
        """Print to both console and buffer"""
        print(*args, **kwargs)
        print(*args, **kwargs, file=self.report_buffer)
    
    def save_report(self, filename: str = 'report.txt'):
        """Save buffered output to file"""
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            f.write(self.report_buffer.getvalue())
        print(f"\nReport saved to {path}")
    
    def save_plot(self, fig, filename: str, dpi: int = 150):
        """Save a figure to file"""
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        self.print(f"Plot saved: {path}")

@dataclass
class Gift:
    """Represents a gift in the exchange"""
    gift_id: int
    true_value: float = 0.0
    appearance_signal: float = 0.0
    is_opened: bool = False
    times_stolen: int = 0  # lifetime steal count (for max_steals_total)
    
    def __hash__(self):
        return hash(self.gift_id)
    
    def __repr__(self):
        status = "opened" if self.is_opened else "wrapped"
        return f"Gift({self.gift_id}, {status}, steals={self.times_stolen})"


@dataclass
class SocialState:
    """Tracks social dynamics for a player"""
    times_stolen_from: Dict[int, int] = field(default_factory=dict)
    times_i_stole_from: Dict[int, int] = field(default_factory=dict)
    total_steals_committed: int = 0
    frustration: float = 0.0
    
    def record_steal(self, victim_id: int):
        self.times_i_stole_from[victim_id] = self.times_i_stole_from.get(victim_id, 0) + 1
        self.total_steals_committed += 1
    
    def record_victimization(self, thief_id: int, frustration_gain: float = 0.15):
        self.times_stolen_from[thief_id] = self.times_stolen_from.get(thief_id, 0) + 1
        self.frustration = min(1.0, self.frustration + frustration_gain)  # Saturates at 1
    
    def decay_frustration(self, amount: float):
        self.frustration = max(0, self.frustration - amount)


class Player:
    """Represents a player with valuation, strategy, and social state"""
    
    def __init__(self, player_id: int, valuations: np.ndarray, config: GameConfig):
        self.player_id = player_id
        self.valuations = valuations
        self.config = config
        self.current_gift: Optional[Gift] = None
        self.seat_position: int = 0
        self.social = SocialState()
        self.observed_values: Dict[int, float] = {}
        self.base_strategy: str = "mean_based"
        self.strategy_func: Callable = None
    
    def true_value_of(self, gift: Gift) -> float:
        """Actual subjective value (omniscient)"""
        return self.valuations[gift.gift_id]
    
    def perceived_value_of(self, gift: Gift) -> float:
        """
        Value estimate given information state.
        
        		for opened gifts: true subjective value;
        		for wrapped gifts: Bayesian posterior mean minus CARA risk penalty.
        """
        if not self.config.enable_partial_information:
            return self.true_value_of(gift)
        
        if gift.is_opened or gift.gift_id in self.observed_values:
            return self.true_value_of(gift)
        
        # Bayesian parameters
        sigma_0_sq = self.config.prior_variance       # prior variance
        sigma_a_sq = self.config.signal_noise ** 2    # signal noise variance
        mu_0 = self.config.prior_mean                 # prior mean
        rho = self.config.risk_aversion               # CARA coefficient
        
        # Bayesian signal weight
        omega = sigma_0_sq / (sigma_0_sq + sigma_a_sq)
        
        # posterior mean:
        posterior_mean = (1 - omega) * mu_0 + omega * gift.appearance_signal
        
        # posterior variance
        posterior_var = (sigma_0_sq * sigma_a_sq) / (sigma_0_sq + sigma_a_sq)
        
        # CARA certainty equivalent
        certainty_equivalent = posterior_mean - (rho / 2) * posterior_var
        
        return max(0, certainty_equivalent)
    
    def social_cost_of_stealing(self, victim: 'Player') -> float:
        """
        Utility cost of stealing from a specific victim
        """
        if not self.config.enable_social_costs:
            return 0.0
        
        base = self.config.steal_base_cost
        prior_steals = self.social.times_i_stole_from.get(victim.player_id, 0)
        repeat_penalty = base * prior_steals * self.config.repeat_steal_multiplier
        reputation_penalty = self.social.total_steals_committed * self.config.reputation_decay
        return base + repeat_penalty + reputation_penalty
    
    def net_value_of_stealing(self, target: 'Player') -> float:
        """
        Expected utility gain from stealing target's gift: value gained minus value lost minus transaction cost.
        """
        if target.current_gift is None:
            return float('-inf')
        
        gift_value = self.perceived_value_of(target.current_gift)
        current_value = self.current_value()
        social_cost = self.social_cost_of_stealing(target)
        return gift_value - current_value - social_cost
    
    def current_value(self) -> float:
        if self.current_gift is None:
            return 0.0
        return self.perceived_value_of(self.current_gift)
    
    def decide_action(self, game_state: 'GameState') -> Tuple[str, Optional['Player']]:
        """Dispatch to strategy with possible adaptation"""
        if self.config.enable_adaptive_strategy:
            return self._adaptive_decide(game_state)
        else:
            return self.strategy_func(self, game_state)
    
    def _adaptive_decide(self, game_state: 'GameState') -> Tuple[str, Optional['Player']]:
        """
        Strategy that adapts based on game state and emotional state
        """
        targets = game_state.get_valid_steal_targets(self)
        unwrapped = game_state.get_unwrapped_gifts()
        
        # game phase
        n = len(game_state.players)
        opened_count = sum(1 for p in game_state.players if p.current_gift is not None)
        phase = opened_count / n
        
        # current satisfaction (perceived value of held gift)
        satisfaction = self.current_value()
        
        # frustration aggression
        frustration_effect = self.social.frustration * self.config.lambda_frustration
        
        # linear approximation to logit choice probability (TODO: CHECK THIS!!!!!)
        base_steal_prob = self._base_steal_probability()
        steal_prob = (base_steal_prob + self.config.lambda_phase * phase + frustration_effect - self.config.lambda_satisfaction * satisfaction)
        steal_prob = np.clip(steal_prob, 0.05, 0.95)  # ε = 0.05 ensures exploration
        
        if random.random() < steal_prob and targets:
            best = max(targets, key=lambda t: self.net_value_of_stealing(t))
            if self.net_value_of_stealing(best) > 0:
                return "StealGift", best
        
        if unwrapped:
            return "OpenGift", None
        
        if targets:
            best = max(targets, key=lambda t: self.net_value_of_stealing(t))
            return "StealGift", best
        
        return "OpenGift", None
    
    def _base_steal_probability(self) -> float:
        """Base aggression level from strategy archetype"""
        probs = {
            "always_open": 0.1,
            "always_steal": 0.9,
            "coin_flip": 0.5,
            "mean_based": 0.6,
            "threshold": 0.5,
        }
        return probs.get(self.base_strategy, 0.5)
    
    def observe_gift(self, gift: Gift):
        """Record that this gift's value is now known"""
        self.observed_values[gift.gift_id] = self.true_value_of(gift)
    
    def __repr__(self):
        return f"Player({self.player_id})"
    
    def __hash__(self):
        return hash(self.player_id)


class GameState:
    """Tracks the state of a single game"""
    
    def __init__(self, players: List[Player], gifts: List[Gift], config: GameConfig):
        self.players = players
        self.gifts = gifts
        self.config = config
        self.round_steal_counts: Dict[int, int] = {}  # gift_id -> steals this round
        self.chain_locked: Set[int] = set()  # gift_ids locked for current chain (prevents ping-pong)
        self.has_taken_primary_turn: Set[Player] = set()
        self.round_number: int = 0
    
    def get_unwrapped_gifts(self) -> List[Gift]:
        return [g for g in self.gifts if not g.is_opened]
    
    def get_opened_gifts(self) -> List[Gift]:
        return [g for g in self.gifts if g.is_opened]
    
    def _gift_is_stealable(self, gift: Gift) -> bool:
        """
        Check if a gift can be stolen given current limits
        We constraint according to:
            1. Cannot steal a gift that was already stolen in this chain (mandatory - prevents A→B→A ping-pong loops)
            2. max_steals_per_round (0 = unlimited across chains)
            3. max_steals_total (0 = unlimited lifetime)
        """
        # chain lock is mandatory (prevents infinite loops!!!!)
        if gift.gift_id in self.chain_locked:
            return False
        
        # check per-round limit (across different chains)
        if self.config.max_steals_per_round > 0:
            round_count = self.round_steal_counts.get(gift.gift_id, 0)
            if round_count >= self.config.max_steals_per_round:
                return False
        
        # check lifetime limit
        if self.config.max_steals_total > 0:
            if gift.times_stolen >= self.config.max_steals_total:
                return False
        
        return True
    
    def get_valid_steal_targets(self, thief: Player) -> List[Player]:
        return [
            p for p in self.players
            if p is not thief
            and p.current_gift is not None
            and self._gift_is_stealable(p.current_gift)
        ]
    
    def steal_gift(self, stealer: Player, victim: Player) -> Player:
        stolen = victim.current_gift
        victim.current_gift = None
        stealer.current_gift = stolen
        
        # update steal counts
        stolen.times_stolen += 1
        self.round_steal_counts[stolen.gift_id] = self.round_steal_counts.get(stolen.gift_id, 0) + 1
        self.chain_locked.add(stolen.gift_id)  # lock for this chain
        
        stealer.social.record_steal(victim.player_id)
        victim.social.record_victimization(stealer.player_id, self.config.frustration_gain)
        return victim
    
    def end_chain(self):
        """Called when a stealing chain ends (someone opens a gift)"""
        self.chain_locked.clear()
    
    def open_gift(self, player: Player, gift: Optional[Gift] = None):
        """Player opens a gift with optional biased selection"""
        unwrapped = self.get_unwrapped_gifts()
        if not unwrapped:
            return
        
        if gift is not None and gift in unwrapped:
            chosen = gift
        elif self.config.enable_biased_selection and len(unwrapped) > 1:
            weights = np.array([np.exp(player.perceived_value_of(g) * 2) for g in unwrapped])
            weights = weights / weights.sum()
            chosen = np.random.choice(unwrapped, p=weights)
        else:
            chosen = random.choice(unwrapped)
        
        chosen.is_opened = True
        player.current_gift = chosen
        
        for p in self.players:
            p.observe_gift(chosen)
        self.end_chain()
    
    def end_round(self):
        """Called at end of each round"""
        self.round_steal_counts.clear()  # Reset per-round steal tracking
        self.chain_locked.clear()  # Should already be clear, but ensure
        self.has_taken_primary_turn.clear()
        self.round_number += 1
        for p in self.players:
            p.social.decay_frustration(self.config.frustration_decay)



# STRATS (TODO: add more? check literature on game-theoretic and pragmatics strategies)


def strategy_always_open(player: Player, game_state: GameState):
    """Always open if possible; else steal randomly"""
    if game_state.get_unwrapped_gifts():
        return "OpenGift", None
    targets = game_state.get_valid_steal_targets(player)
    if targets:
        return "StealGift", random.choice(targets)
    return "OpenGift", None


def strategy_always_steal(player: Player, game_state: GameState):
    """Always steal best available gift"""
    targets = game_state.get_valid_steal_targets(player)
    if not targets:
        return "OpenGift", None
    best = max(targets, key=lambda t: player.net_value_of_stealing(t))
    return "StealGift", best


def strategy_coin_flip(player: Player, game_state: GameState):
    """50/50 steal vs open"""
    if random.random() < 0.5:
        targets = game_state.get_valid_steal_targets(player)
        if targets:
            best = max(targets, key=lambda t: player.net_value_of_stealing(t))
            return "StealGift", best
    if game_state.get_unwrapped_gifts():
        return "OpenGift", None
    targets = game_state.get_valid_steal_targets(player)
    if targets:
        best = max(targets, key=lambda t: player.net_value_of_stealing(t))
        return "StealGift", best
    return "OpenGift", None


def strategy_mean_based(player: Player, game_state: GameState):
    """Steal if gift above mean of opened gifts"""
    opened = game_state.get_opened_gifts()
    if not opened:
        return "OpenGift", None
    
    mean_val = statistics.mean(player.perceived_value_of(g) for g in opened)
    targets = game_state.get_valid_steal_targets(player)
    above_mean = [t for t in targets
                  if player.net_value_of_stealing(t) > mean_val - player.current_value()]
    
    if above_mean:
        best = max(above_mean, key=lambda t: player.net_value_of_stealing(t))
        return "StealGift", best
    if game_state.get_unwrapped_gifts():
        return "OpenGift", None
    if targets:
        best = max(targets, key=lambda t: player.net_value_of_stealing(t))
        return "StealGift", best
    return "OpenGift", None


def strategy_threshold(player: Player, game_state: GameState):
    """Only steal if gift exceeds threshold"""
    threshold = 0.6
    targets = game_state.get_valid_steal_targets(player)
    good = [t for t in targets if player.net_value_of_stealing(t) > threshold]
    
    if good:
        best = max(good, key=lambda t: player.net_value_of_stealing(t))
        return "StealGift", best
    if game_state.get_unwrapped_gifts():
        return "OpenGift", None
    if targets:
        best = max(targets, key=lambda t: player.net_value_of_stealing(t))
        return "StealGift", best
    return "OpenGift", None

def strategy_expected_value(player: Player, game_state: GameState):
    """Steal only if best steal exceeds expected value of opening"""
    unwrapped = game_state.get_unwrapped_gifts()
    targets = game_state.get_valid_steal_targets(player)
    
    # expected value of opening (uses PI if enabled)
    if unwrapped:
        ev_open = np.mean([player.perceived_value_of(g) for g in unwrapped])
    else:
        ev_open = 0
    
    if targets:
        best_target = max(targets, key=lambda t: player.net_value_of_stealing(t))
        ev_steal = player.net_value_of_stealing(best_target)
        
        if ev_steal > ev_open:
            return "StealGift", best_target
    
    if unwrapped:
        return "OpenGift", None
    if targets:
        return "StealGift", max(targets, key=lambda t: player.net_value_of_stealing(t))
    return "OpenGift", None


STRATEGIES = {
    "always_open": strategy_always_open,
    "always_steal": strategy_always_steal,
    "coin_flip": strategy_coin_flip,
    "mean_based": strategy_mean_based,
    "threshold": strategy_threshold,
    "expected_value": strategy_expected_value,
}



# MECHANICS (see paper)


def generate_gifts(n: int, config: GameConfig) -> Tuple[List[Gift], np.ndarray]:
    """Generate gifts with valuations based on model type"""
    model = config.valuation_model
    
    if model == 'correlated':
        common = np.random.rand(n)
        rho = config.correlation_strength
        noise = np.random.rand(n, n)
        V = rho * common + np.sqrt(1 - rho**2) * noise
        V = np.clip(V, 0, 1)
        objective_quality = common
    elif model == 'independent':
        V = np.random.rand(n, n)
        objective_quality = np.mean(V, axis=0)
    else:  # neg_correlated
        base = np.random.rand(n)
        V = np.zeros((n, n))
        for i in range(n):
            if i % 2 == 0:
                V[i] = base
            else:
                V[i] = 1.0 - base
            V[i] += np.random.randn(n) * 0.2
        V = np.clip(V, 0, 1)
        objective_quality = base
    
    gifts = []
    for i in range(n):
        appearance = np.clip(objective_quality[i] + np.random.randn() * config.signal_noise, 0, 1)
        gifts.append(Gift(gift_id=i, true_value=objective_quality[i], appearance_signal=appearance))
    
    return gifts, V


def stealing_chain(displaced: Player, game_state: GameState):
    """Handle chain of steals after initial steal (iterative)"""
    while True:
        action, target = displaced.decide_action(game_state)
        
        if action == "OpenGift":
            game_state.open_gift(displaced)
            return
        
        valid = game_state.get_valid_steal_targets(displaced)
        if not valid:
            game_state.open_gift(displaced)
            return
        
        if target not in valid:
            target = max(valid, key=lambda t: displaced.net_value_of_stealing(t))
        
        displaced = game_state.steal_gift(displaced, target)


def primary_turn(player: Player, game_state: GameState):
    """Execute a player's primary turn"""
    game_state.has_taken_primary_turn.add(player)
    action, target = player.decide_action(game_state)
    
    if action == "OpenGift":
        game_state.open_gift(player)
        return
    
    valid = game_state.get_valid_steal_targets(player)
    if not valid:
        game_state.open_gift(player)
        return
    
    if target not in valid:
        target = max(valid, key=lambda t: player.net_value_of_stealing(t))
    
    displaced = game_state.steal_gift(player, target)
    stealing_chain(displaced, game_state)


def final_swap_option(game_state: GameState):
    """Player 1 may swap with anyone at the end"""
    p1 = game_state.players[0]
    if not p1.current_gift:
        return
    
    others = [p for p in game_state.players[1:] if p.current_gift]
    if not others:
        return
    
    best = max(others, key=lambda p: p1.net_value_of_stealing(p))
    if p1.net_value_of_stealing(best) > 0:
        p1.current_gift, best.current_gift = best.current_gift, p1.current_gift
        p1.social.record_steal(best.player_id)
        best.social.record_victimization(p1.player_id, game_state.config.frustration_gain)


def play_one_game(config: GameConfig) -> dict:
    """Simulate a single game and return results"""
    n = config.num_players
    gifts, V = generate_gifts(n, config)
    
    players = []
    strategy_names = list(STRATEGIES.keys())
    for i in range(n):
        strat_name = random.choice(strategy_names)
        p = Player(i, V[i], config)
        p.base_strategy = strat_name
        p.strategy_func = STRATEGIES[strat_name]
        players.append(p)
    
    random.shuffle(players)
    for seat, p in enumerate(players):
        p.seat_position = seat + 1
    
    game_state = GameState(players, gifts, config)
    
    total_steals = 0
    chain_lengths = []
    
    for round_k in range(n):
        current = players[round_k]
        if current not in game_state.has_taken_primary_turn:
            steals_before = sum(p.social.total_steals_committed for p in players)
            primary_turn(current, game_state)
            steals_after = sum(p.social.total_steals_committed for p in players)
            chain_len = steals_after - steals_before
            if chain_len > 0:
                chain_lengths.append(chain_len)
            total_steals += chain_len
        game_state.end_round()
    
    final_swap_option(game_state)
    
    results = {
        'player_data': [],
        'total_steals': total_steals,
        'chain_lengths': chain_lengths,
    }
    
    for p in players:
        final_value = p.true_value_of(p.current_gift) if p.current_gift else 0.0
        results['player_data'].append({
            'strategy': p.base_strategy,
            'seat': p.seat_position,
            'final_value': final_value,
            'steals_committed': p.social.total_steals_committed,
            'final_frustration': p.social.frustration,
        })
    
    return results


def run_simulation(config: GameConfig, desc: str = "Simulating") -> dict:
    """Run full simulation and return aggregated results"""
    strat_values = defaultdict(list)
    seat_values = defaultdict(list)
    all_chain_lengths = []
    total_steals_per_game = []
    frustration_by_seat = defaultdict(list)
    
    for _ in tqdm(range(config.num_games), desc=desc):
        result = play_one_game(config)
        total_steals_per_game.append(result['total_steals'])
        all_chain_lengths.extend(result['chain_lengths'])
        
        for pd in result['player_data']:
            strat_values[pd['strategy']].append(pd['final_value'])
            seat_values[pd['seat']].append(pd['final_value'])
            frustration_by_seat[pd['seat']].append(pd['final_frustration'])
    
    return {
        'strat_values': dict(strat_values),
        'seat_values': dict(seat_values),
        'chain_lengths': all_chain_lengths,
        'steals_per_game': total_steals_per_game,
        'frustration_by_seat': dict(frustration_by_seat),
        'config': config,
    }



# EXPERIMENT


def run_single_experiment(features: List[str], num_players: int, num_games: int,
                          seed: int, valuation_model: str, 
                          output: OutputManager,
                          max_steals_per_round: int = 1,
                          max_steals_total: int = 0) -> Tuple[GameConfig, dict]:
    """Run simulation with specified features"""
    config = config_from_features(features, num_players, num_games, seed, valuation_model,
                                  max_steals_per_round, max_steals_total)
    label = config_label(config)
    
    output.print(f"\n{'='*60}")
    output.print(f"running: {label}")
    output.print(f"feature set: {features if features else ['none (base game)']}")
    output.print(f"valuation model: {valuation_model}")
    output.print(f"steal limits: per_round={max_steals_per_round}, total={max_steals_total}")
    output.print(f"{'='*60}")
    
    np.random.seed(seed)
    random.seed(seed)
    results = run_simulation(config, desc=label)
    
    return config, results


def run_isolation_study(num_players: int, num_games: int, seed: int,
                        valuation_model: str, 
                        output: OutputManager,
                        max_steals_per_round: int = 1,
                        max_steals_total: int = 0) -> Dict[str, Tuple[GameConfig, dict]]:
    """Compare BASE vs each single feature vs FULL"""
    experiments = {
        'BASE': [],
        '+partial_info': ['partial_info'],
        '+social_costs': ['social_costs'],
        '+adaptive': ['adaptive'],
        '+biased_selection': ['biased_selection'],
        'FULL': FEATURES.copy(),
    }
    
    results = {}
    for name, features in experiments.items():
        np.random.seed(seed)
        random.seed(seed)
        config, res = run_single_experiment(features, num_players, num_games, 
                                            seed, valuation_model, output,
                                            max_steals_per_round, max_steals_total)
        results[name] = (config, res)
    
    return results


def run_ablation_study(num_players: int, num_games: int, seed: int,
                       valuation_models: List[str],
                       output: OutputManager,
                       max_steals_per_round: int = 1,
                       max_steals_total: int = 0) -> Dict[str, Dict[str, Tuple[GameConfig, dict]]]:
    """Run all 16 combinations of features across specified valuation models"""
    all_results = {}
    
    for model in valuation_models:
        output.print(f"\n{'#'*70}")
        output.print(f"# ABLATION STUDY: valuation Model = {model.upper()}")
        output.print(f"{'#'*70}")
        
        model_results = {}
        for r in range(5):
            for combo in combinations(FEATURES, r):
                features = list(combo)
                np.random.seed(seed)
                random.seed(seed)
                config, results = run_single_experiment(features, num_players, num_games, 
                                                        seed, model, output,
                                                        max_steals_per_round, max_steals_total)
                key = config_label(config)
                model_results[key] = (config, results)
        
        all_results[model] = model_results
    
    return all_results


def run_additive_study(num_players: int, num_games: int, seed: int,
                       valuation_model: str,
                       output: OutputManager,
                       max_steals_per_round: int = 1,
                       max_steals_total: int = 0) -> Dict[str, Tuple[GameConfig, dict]]:
    """Add features incrementally"""
    stages = [
        ('BASE', []),
        ('+PI', ['partial_info']),
        ('+PI+SC', ['partial_info', 'social_costs']),
        ('+PI+SC+AD', ['partial_info', 'social_costs', 'adaptive']),
        ('FULL', FEATURES.copy()),
    ]
    
    results = {}
    for name, features in stages:
        np.random.seed(seed)
        random.seed(seed)
        config, res = run_single_experiment(features, num_players, num_games, 
                                            seed, valuation_model, output,
                                            max_steals_per_round, max_steals_total)
        results[name] = (config, res)
    
    return results



# Output and metrics


def compute_summary_stats(results: dict, num_players: int) -> dict:
    """Extract key metrics from results"""
    stats = {
        'steals_per_game': np.mean(results['steals_per_game']),
        'steals_std': np.std(results['steals_per_game']),
        'avg_chain_length': np.mean(results['chain_lengths']) if results['chain_lengths'] else 0,
        'max_chain_length': max(results['chain_lengths']) if results['chain_lengths'] else 0,
        'seat_1_value': np.mean(results['seat_values'].get(1, [0])),
        'seat_2_value': np.mean(results['seat_values'].get(2, [0])),
        'seat_last_value': np.mean(results['seat_values'].get(num_players, [0])),
    }
    
    for strat in STRATEGIES.keys():
        if strat in results['strat_values']:
            stats[f'strat_{strat}'] = np.mean(results['strat_values'][strat])
    
    return stats


def print_comparison_table(all_results: Dict[str, Tuple[GameConfig, dict]], 
                           num_players: int, output: OutputManager,
                           model_name: str = None):
    """Print formatted comparison table"""
    header_suffix = f" [{model_name}]" if model_name else ""
    
    output.print("\n" + "=" * 95)
    output.print(f"FEATURE EFFECT COMPARISON{header_suffix}")
    output.print("=" * 95)
    
    metrics = ['steals_per_game', 'avg_chain_length', 'seat_1_value', 'seat_2_value', 'seat_last_value']
    metric_labels = ['Steals/Game', 'Chain length', 'Seat 1', 'Seat 2', f'Seat {num_players}']
    
    header = f"{'Configuration':<20}" + "".join(f"{m:<14}" for m in metric_labels)
    output.print(header)
    output.print("-" * 95)
    
    for name, (config, results) in all_results.items():
        stats = compute_summary_stats(results, num_players)
        row = f"{name:<20}"
        for metric in metrics:
            row += f"{stats[metric]:<14.3f}"
        output.print(row)
    
    output.print("\n" + "=" * 95)
    output.print(f"STRATEGY PERFORMANCE BY CONFIGURATION{header_suffix}")
    output.print("=" * 95)
    
    strat_names = list(STRATEGIES.keys())
    header = f"{'Configuration':<20}" + "".join(f"{s:<14}" for s in strat_names)
    output.print(header)
    output.print("-" * 95)
    
    for name, (config, results) in all_results.items():
        row = f"{name:<20}"
        for strat in strat_names:
            val = np.mean(results['strat_values'].get(strat, [0]))
            row += f"{val:<14.4f}"
        output.print(row)


def compute_feature_effects(isolation_results: Dict[str, Tuple[GameConfig, dict]],
                            num_players: int) -> dict:
    """Compute marginal effect of each feature relative to BASE"""
    base_stats = compute_summary_stats(isolation_results['BASE'][1], num_players)
    
    effects = {}
    for name, (config, results) in isolation_results.items():
        if name == 'BASE':
            continue
        stats = compute_summary_stats(results, num_players)
        effects[name] = {metric: stats[metric] - base_stats[metric] for metric in base_stats.keys()}
    
    return effects


def print_feature_effects(effects: dict, output: OutputManager):
    """Print marginal effects table"""
    output.print("\n" + "=" * 95)
    output.print("MARGINAL FEATURE EFFECTS (relative to BASE)")
    output.print("=" * 95)
    
    metrics = ['steals_per_game', 'avg_chain_length', 'seat_1_value']
    metric_labels = ['Δ Steals/Game', 'Δ Chain Length', 'Δ Seat 1 Value']
    
    header = f"{'Feature':<25}" + "".join(f"{m:<20}" for m in metric_labels)
    output.print(header)
    output.print("-" * 95)
    
    for name, deltas in effects.items():
        if name == 'FULL':
            continue
        row = f"{name:<25}"
        for metric in metrics:
            val = deltas[metric]
            sign = '+' if val >= 0 else ''
            row += f"{sign}{val:<19.3f}"
        output.print(row)
    
    if 'FULL' in effects:
        output.print("-" * 95)
        individual_sum = {metric: sum(effects[n][metric] for n in effects if n != 'FULL') for metric in metrics}
        interaction = {metric: effects['FULL'][metric] - individual_sum[metric] for metric in metrics}
        
        row = f"{'FULL (total)':<25}"
        for metric in metrics:
            val = effects['FULL'][metric]
            sign = '+' if val >= 0 else ''
            row += f"{sign}{val:<19.3f}"
        output.print(row)
        
        row = f"{'interaction effect':<25}"
        for metric in metrics:
            val = interaction[metric]
            sign = '+' if val >= 0 else ''
            row += f"{sign}{val:<19.3f}"
        output.print(row)



# PLOTS


def plot_steals_per_game(all_results: Dict[str, Tuple[GameConfig, dict]], 
                         num_players: int, output: OutputManager,
                         prefix: str = ''):
    """Plot steals per game bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(all_results.keys())
    summaries = {name: compute_summary_stats(res, num_players) for name, (_, res) in all_results.items()}
    
    vals = [summaries[n]['steals_per_game'] for n in names]
    stds = [summaries[n]['steals_std'] for n in names]
    
    bars = ax.bar(range(len(names)), vals, color='steelblue', alpha=0.7, yerr=stds, capsize=3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('steals per game')
    ax.set_title('stealing frequency by configuration')
    ax.axhline(y=vals[0], color='gray', linestyle='--', alpha=0.5, label='BASE reference')
    ax.legend()
    
    plt.tight_layout()
    filename = f"{prefix}steals_per_game.png" if prefix else "steals_per_game.png"
    output.save_plot(fig, filename)


def plot_chain_length(all_results: Dict[str, Tuple[GameConfig, dict]], 
                      num_players: int, output: OutputManager,
                      prefix: str = ''):
    """Plot chain length bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(all_results.keys())
    summaries = {name: compute_summary_stats(res, num_players) for name, (_, res) in all_results.items()}
    
    avg_vals = [summaries[n]['avg_chain_length'] for n in names]
    max_vals = [summaries[n]['max_chain_length'] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax.bar(x - width/2, avg_vals, width, label='average', color='forestgreen', alpha=0.7)
    ax.bar(x + width/2, max_vals, width, label='maximum', color='darkgreen', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('chain length')
    ax.set_title('stealing chain length by configuration')
    ax.legend()
    
    plt.tight_layout()
    filename = f"{prefix}chain_length.png" if prefix else "chain_length.png"
    output.save_plot(fig, filename)


def plot_seat_advantage(all_results: Dict[str, Tuple[GameConfig, dict]], 
                        num_players: int, output: OutputManager,
                        prefix: str = ''):
    """Plot seat 1 vs seat 2 advantage"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(all_results.keys())
    summaries = {name: compute_summary_stats(res, num_players) for name, (_, res) in all_results.items()}
    
    seat1 = [summaries[n]['seat_1_value'] for n in names]
    seat2 = [summaries[n]['seat_2_value'] for n in names]
    seat_last = [summaries[n]['seat_last_value'] for n in names]
    
    x = np.arange(len(names))
    width = 0.25
    
    ax.bar(x - width, seat1, width, label='Seat 1 (first and final swap)', color='gold', alpha=0.8)
    ax.bar(x, seat2, width, label='Seat 2 (worst position)', color='indianred', alpha=0.8)
    ax.bar(x + width, seat_last, width, label=f'Seat {num_players} (last)', color='steelblue', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('mean final value')
    ax.set_title('seat position advantage')
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    filename = f"{prefix}seat_advantage.png" if prefix else "seat_advantage.png"
    output.save_plot(fig, filename)


def plot_strategy_performance(all_results: Dict[str, Tuple[GameConfig, dict]], 
                              num_players: int, output: OutputManager,
                              prefix: str = ''):
    """Plot aggressive vs passive strategy performance"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(all_results.keys())
    summaries = {name: compute_summary_stats(res, num_players) for name, (_, res) in all_results.items()}
    
    aggressive = [np.mean([summaries[n].get(f'strat_{s}', 0)
                          for s in ['always_steal', 'mean_based', 'threshold']])
                  for n in names]
    passive = [np.mean([summaries[n].get(f'strat_{s}', 0)
                       for s in ['always_open', 'coin_flip']])
               for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax.bar(x - width/2, aggressive, width, label='Aggressive (steal/mean/threshold)', 
           alpha=0.7, color='indianred')
    ax.bar(x + width/2, passive, width, label='Passive (open/flip)', 
           alpha=0.7, color='mediumseagreen')
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('mean final value')
    ax.set_title('aggressive versus passive strategy performance')
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    filename = f"{prefix}strategy_performance.png" if prefix else "strategy_performance.png"
    output.save_plot(fig, filename)


def plot_strategy_ranking(all_results: Dict[str, Tuple[GameConfig, dict]], 
                          num_players: int, output: OutputManager,
                          prefix: str = ''):
    """Plot individual strategy rankings across configurations"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = list(all_results.keys())
    strat_names = list(STRATEGIES.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(strat_names)))
    
    x = np.arange(len(names))
    width = 0.15
    
    for i, strat in enumerate(strat_names):
        vals = [np.mean(all_results[n][1]['strat_values'].get(strat, [0])) for n in names]
        ax.bar(x + i * width - (len(strat_names) - 1) * width / 2, vals, width, 
               label=strat, color=colors[i], alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('mean final value')
    ax.set_title('strategy performance breakdown')
    ax.legend(loc='upper right', ncol=2)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    filename = f"{prefix}strategy_ranking.png" if prefix else "strategy_ranking.png"
    output.save_plot(fig, filename)


def plot_seat_heatmap(all_results: Dict[str, Tuple[GameConfig, dict]], 
                      num_players: int, output: OutputManager,
                      prefix: str = ''):
    """Plot heatmap of seat values across configurations"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    names = list(all_results.keys())
    seats = list(range(1, num_players + 1))
    
    # Build matrix
    matrix = np.zeros((len(names), len(seats)))
    for i, name in enumerate(names):
        results = all_results[name][1]
        for j, seat in enumerate(seats):
            matrix[i, j] = np.mean(results['seat_values'].get(seat, [0]))
    
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0.4, vmax=1.0)
    
    ax.set_xticks(range(len(seats)))
    ax.set_xticklabels(seats)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('seat position')
    ax.set_ylabel('configuration')
    ax.set_title('mean final value by seat position and configuration')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('mean final value')
    
    plt.tight_layout()
    filename = f"{prefix}seat_heatmap.png" if prefix else "seat_heatmap.png"
    output.save_plot(fig, filename)


def generate_all_plots(all_results: Dict[str, Tuple[GameConfig, dict]], 
                       num_players: int, output: OutputManager,
                       prefix: str = ''):
    """Generate all individual plots"""
    plot_steals_per_game(all_results, num_players, output, prefix)
    plot_chain_length(all_results, num_players, output, prefix)
    plot_seat_advantage(all_results, num_players, output, prefix)
    plot_strategy_performance(all_results, num_players, output, prefix)
    plot_strategy_ranking(all_results, num_players, output, prefix)
    
    # Only generate heatmap if not too many configurations
    if len(all_results) <= 16:
        plot_seat_heatmap(all_results, num_players, output, prefix)



# usual main


def main():
    parser = argparse.ArgumentParser(
        description='White Elephant Gift Exchange Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python white_elephant.py --features none                  # base game only
  python white_elephant.py --features all                   # fully decorated
  python white_elephant.py --features social_costs          # single feature
  python white_elephant.py --isolation                      # feature isolation study
  python white_elephant.py --ablation                       # all 16 combinations (correlated)
  python white_elephant.py --ablation --valuation-model all # all 48 combinations
  python white_elephant.py --additive                       # incremental features
  python white_elephant.py --output-dir ./my_results        # some custom output directory
        """
    )
    
    parser.add_argument('--features', nargs='*', default=None,
                        help='features: partial_info, social_costs, adaptive, biased_selection, all, none')
    parser.add_argument('--isolation', action='store_true',
                        help='run feature isolation study')
    parser.add_argument('--ablation', action='store_true',
                        help='run full ablation study (16 combinations per valuation model)')
    parser.add_argument('--additive', action='store_true',
                        help='run additive study (features added incrementally)')
    parser.add_argument('--valuation-model', nargs='+', default=['correlated'],
                        help='valuation model(s): independent, correlated, neg_correlated, all')
    parser.add_argument('--num-players', type=int, default=29,
                        help='number of players (default: 29)')
    parser.add_argument('--num-games', type=int, default=5000,
                        help='number of games (default: 5000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='output directory for reports and plots')
    parser.add_argument('--max-steals-per-round', type=int, default=1,
                        help='max times a gift can be stolen per round; 0=unlimited (default: 1)')
    parser.add_argument('--max-steals-total', type=int, default=0,
                        help='max times a gift can be stolen total; 0=unlimited (default: 0)')
    
    args = parser.parse_args()
    
    # resolve valuation models
    if 'all' in args.valuation_model:
        valuation_models = VALUATION_MODELS.copy()
    else:
        valuation_models = [m for m in args.valuation_model if m in VALUATION_MODELS]
        invalid = [m for m in args.valuation_model if m not in VALUATION_MODELS and m != 'all']
        if invalid:
            print(f"Warning: unknown valuation models ignored: {invalid}")
        if not valuation_models:
            valuation_models = ['correlated']
    
    # Create output manager
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    output = OutputManager(output_dir)
    
    output.print(f"gift exchange simulation")
    output.print(f"run timestamp: {timestamp}")
    output.print(f"output directory: {output_dir}")
    output.print(f"parameters: n={args.num_players}, games={args.num_games}, seed={args.seed}")
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Extract steal limit settings
    steal_per_round = args.max_steals_per_round
    steal_total = args.max_steals_total
    
    if args.ablation:
        output.print(f"\nrunning FULL ABLATION STUDY across {len(valuation_models)} valuation model(s)...")
        output.print(f"models: {valuation_models}")
        output.print(f"total configurations: {16 * len(valuation_models)}")
        output.print(f"steal limits: per_round={steal_per_round}, total={steal_total}")
        
        all_model_results = run_ablation_study(args.num_players, args.num_games, 
                                                args.seed, valuation_models, output,
                                                steal_per_round, steal_total)
        
        for model, results in all_model_results.items():
            output.print(f"\n{'='*95}")
            output.print(f"RESULTS FOR VALUATION MODEL: {model.upper()}")
            output.print(f"{'='*95}")
            print_comparison_table(results, args.num_players, output, model)
            
            # Generate plots with model prefix
            prefix = f"{model}_"
            generate_all_plots(results, args.num_players, output, prefix)
        
    elif args.isolation:
        model = valuation_models[0]
        output.print(f"\nrunning FEATURE ISOLATION STUDY with model={model}...")
        output.print(f"steal limits: per_round={steal_per_round}, total={steal_total}")
        
        results = run_isolation_study(args.num_players, args.num_games, 
                                      args.seed, model, output,
                                      steal_per_round, steal_total)
        print_comparison_table(results, args.num_players, output)
        effects = compute_feature_effects(results, args.num_players)
        print_feature_effects(effects, output)
        generate_all_plots(results, args.num_players, output)
        
    elif args.additive:
        model = valuation_models[0]
        output.print(f"\nrunning ADDITIVE FEATURE STUDY with model={model}...")
        output.print(f"steal limits: per_round={steal_per_round}, total={steal_total}")
        
        results = run_additive_study(args.num_players, args.num_games, 
                                     args.seed, model, output,
                                     steal_per_round, steal_total)
        print_comparison_table(results, args.num_players, output)
        generate_all_plots(results, args.num_players, output)
        
    elif args.features is not None:
        model = valuation_models[0]
        
        if 'all' in args.features:
            features = FEATURES.copy()
        elif 'none' in args.features:
            features = []
        else:
            features = [f for f in args.features if f in FEATURES]
            invalid = [f for f in args.features if f not in FEATURES and f not in ['all', 'none']]
            if invalid:
                output.print(f"Warning: unknown features ignored: {invalid}")
        
        config, results = run_single_experiment(
            features, args.num_players, args.num_games, 
            args.seed, model, output,
            steal_per_round, steal_total
        )
        
        stats = compute_summary_stats(results, args.num_players)
        output.print(f"\n{'metric':<30} {'value':<15}")
        output.print("-" * 45)
        for metric, val in stats.items():
            output.print(f"{metric:<30} {val:<15.4f}")
        
        # for single experiment, create minimal plots
        single_results = {config_label(config): (config, results)}
        plot_strategy_ranking(single_results, args.num_players, output)
    
    else:
        model = valuation_models[0]
        output.print(f"\nno mode specified; running isolation study with model={model}...")
        output.print(f"steal limits: per_round={steal_per_round}, total={steal_total}")
        
        results = run_isolation_study(args.num_players, args.num_games, 
                                      args.seed, model, output,
                                      steal_per_round, steal_total)
        print_comparison_table(results, args.num_players, output)
        effects = compute_feature_effects(results, args.num_players)
        print_feature_effects(effects, output)
        generate_all_plots(results, args.num_players, output)
    
    output.save_report('report.txt')
    output.print(f"\n{'='*60}")
    output.print(f"all outputs saved to: {output_dir}")
    output.print(f"{'='*60}")


if __name__ == "__main__":
    main()
