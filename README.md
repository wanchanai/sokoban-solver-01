# sokoban-solver-01

A push-optimal Sokoban solver for the make.fun bounty
[sokoban-solver-01-run-20261121](https://make.fun/bounty/sokoban-solver-01-run-20261121).

![bounty](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)
![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/wanchanai/sokoban-solver-01/rank.svg)
![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/wanchanai/sokoban-solver-01/reward.svg)

## What it does

Each level is a one-square-wide hallway with small rooms (at most 6x6,
3 or 4 boxes each) opening off it. A box pushed out of its room can never
come back, so every room is an independent subproblem and a level's
push optimum is the sum of its rooms' optima.

The solver:

1. Splits the level into rooms by cutting the dead squares of the
   player's hallway row (squares from which no box can reach a goal,
   found by pulling boxes backwards from the goals over the walls alone).
2. Solves each room in turn with a breadth-first search over pushes —
   a state is the sorted box squares plus the player's reachable region —
   which is push-optimal by construction. Dead-square pruning and
   frozen 2x2-block pruning keep the search small, and the other rooms'
   boxes stand as walls while a room is solved.
3. Falls back to a push-based A* (minimum-cost box-to-goal matching as
   the estimate) for any room that ever exceeds the BFS state budget.

All budgets are counted in states, never in time, and every tie is
broken by a fixed order, so runs are byte-for-byte deterministic.

On the kit's practice set it scores 1,000,000 / 1,000,000 (every level
solved in the optimal number of pushes).

## Run it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<dir with i001..i005/level.txt> MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, no dependencies. Answers are written to
`out/<id>/solution.txt` as LURD move strings.
