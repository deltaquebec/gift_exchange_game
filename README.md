# Formal specification and behavioral simulation of the holiday gift exchange game

This repository accompanies the paper:

> **Formal specification and behavioral simulation of the holiday gift exchange game**
> Daniel Quigley

## Overview

This repository contains:
- Paper formalizing game mechanics, proving termination and bijection properties, and deriving trajectory counts;
- Python simulation framework implementing the base and decorated game variants

## Repository contents
```
├── paper/
│   └── main.pdf
├── src/
│   ├── game.py          # game implementation
│   └── trajectory.py    # counting algorithm
└── README.md
```

## Contributions

1. Complete specification of players, gifts, ownership, stealing chains, and configurable steal limits. Proof that chains terminate in at most $n-1$ steals; proof that final allocations are bijective.

2. Extensions incorporating partial information (Bayesian posteriors over wrapped gifts), social costs (norm violation, relationship damage, reputation), adaptive dynamics (frustration/satisfaction feedback), and biased selection (appearance-weighted choice).

3. Full factorial design crossing 4 binary features × 3 valuation models × 6 strategies across 240,000 games. Social costs reduce stealing 27–48%; partial information contributes minimally.

4. Derivation connecting game path counts to OEIS A000522; algorithm for exact counts under finite lifetime steal limits.

## Usage
```bash
# install dependencies
pip install numpy pandas scipy

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
```

## Citation
```bibtex
@article{quigley2025giftexchange,
  title={Formal specification and behavioral simulation of the holiday gift exchange game},
  author={Quigley, Daniel},
  year={2025}
}
```
