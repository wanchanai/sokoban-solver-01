// Sokoban solver for sokoban-solver-01 (make.fun).
//
// Strategy
// --------
// Every level is a one-square-wide hallway with small rooms (at most 6x6,
// 3 or 4 boxes each) opening off it. A box pushed out of its room can never
// come back, so each room is an independent push-optimal subproblem and a
// level's optimum is the sum of its rooms' optima.
//
// 1. Split the level into rooms: squares of the player's hallway row from
//    which no box can ever reach a goal ("dead" squares, found by pulling
//    boxes backwards from the goals over the walls alone) are treated as
//    cuts; each remaining connected part that holds boxes is one room.
// 2. Solve each room in turn with a breadth-first search over pushes
//    (a state is the sorted box squares plus the player's reachable
//    region, named by its lowest square). BFS over pushes visits states
//    in push order, so the first solution found is push-optimal.
//    Two sound prunes keep the search small: a box may only be pushed
//    onto a live square, and a push that freezes a 2x2 block of walls
//    and boxes with a box off its goal is dropped.
// 3. If a room ever exceeds the BFS state budget, fall back to a
//    push-based A* for that room whose estimate is a minimum-cost
//    matching of boxes to goals by grid distance (admissible, so the
//    fallback is still push-optimal when it finishes; it usually
//    expands far fewer states than BFS on open rooms).
// 4. If the level does not split into rooms at all, run the same
//    push A* on the whole level with a large deterministic budget.
//
// Everything is budgeted in states, never in time, and every tie is
// broken by fixed square/direction order, so two runs of this program
// on the same input write byte-identical outputs.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Directions in a fixed order: [dx, dy, walk letter, push letter]. */
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

/** Deterministic state budgets (work, never the clock). */
const ROOM_BFS_BUDGET = 450_000;
const ROOM_ASTAR_BUDGET = 900_000;
const WHOLE_ASTAR_BUDGET = 1_500_000;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outer frame is wall whatever the text shows.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/**
 * Live squares: those from which a single box, ignoring the others, can
 * be pushed onto some goal. Found backwards: from a live square x, a box
 * could have been pushed there from y = x - d while the player stood on
 * y - d, so y is live when both y and y - d are open floor.
 */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue: number[] = [];
  for (const g of goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** Scratch flood-fill machinery shared by one solve (stamp technique). */
class Reach {
  seen: Int32Array;
  stack: Int32Array;
  stamp = 0;
  constructor(size: number) {
    this.seen = new Int32Array(size);
    this.stack = new Int32Array(size);
  }
  /** Flood from `from` around walls and `occupied`; returns the lowest square reached. Marks use this.stamp. */
  flood(w: number, wall: Uint8Array, occupied: Uint8Array, from: number): number {
    this.stamp++;
    const stamp = this.stamp;
    const { seen, stack } = this;
    let n = 0;
    let low = from;
    stack[n++] = from;
    seen[from] = stamp;
    while (n > 0) {
      const p = stack[--n]!;
      if (p < low) low = p;
      const up = p - w;
      const down = p + w;
      const left = p - 1;
      const right = p + 1;
      if (!wall[up] && !occupied[up] && seen[up] !== stamp) { seen[up] = stamp; stack[n++] = up; }
      if (!wall[down] && !occupied[down] && seen[down] !== stamp) { seen[down] = stamp; stack[n++] = down; }
      if (!wall[left] && !occupied[left] && seen[left] !== stamp) { seen[left] = stamp; stack[n++] = left; }
      if (!wall[right] && !occupied[right] && seen[right] !== stamp) { seen[right] = stamp; stack[n++] = right; }
    }
    return low;
  }
  has(square: number): boolean {
    return this.seen[square] === this.stamp;
  }
}

/** Shortest walk from `from` to `to` avoiding walls and boxes, as lower-case letters ('' if already there, null if unreachable). */
function walkPath(level: Level, occupied: Uint8Array, from: number, to: number): string | null {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue: number[] = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** Whether the box just pushed onto `t` is part of a frozen 2x2 block holding a box off its goal. */
function frozenBlock(level: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

interface SubProblem {
  level: Level; // board with foreign boxes baked in as walls
  live: Uint8Array;
  goal: Uint8Array;
}

/** A numeric state key: boxes and the region's lowest square in base `size` (exact for the boards and box counts here). */
function stateKey(boxes: number[], low: number, size: number): number {
  let k = low;
  for (const b of boxes) k = k * size + b;
  return k;
}

interface Solved {
  pushes: number;
  moves: string;
}

/**
 * Push-optimal breadth-first search over pushes for one subproblem.
 * Returns null when the state budget runs out or no solution exists.
 */
function solveBfs(sub: SubProblem, startBoxes: number[], startPlayer: number, budget: number): Solved | null {
  const { level, live, goal } = sub;
  const { w, wall } = level;
  const size = wall.length;
  if (startBoxes.some((b) => !live[b])) return null;
  const isSolved = (bs: number[]): boolean => bs.every((b) => goal[b] === 1);
  if (isSolved(startBoxes)) return { pushes: 0, moves: '' };

  const occupied = new Uint8Array(size);
  const region = new Reach(size);
  const keyReach = new Reach(size);
  const boxesList: number[][] = [startBoxes.slice()];
  const playerList: number[] = [startPlayer];
  const parent: number[] = [-1];
  const segment: string[] = [''];
  const visited = new Set<number>();
  for (const b of startBoxes) occupied[b] = 1;
  visited.add(stateKey(startBoxes, keyReach.flood(w, wall, occupied, startPlayer), size));
  occupied.fill(0);

  let found = -1;
  for (let head = 0; head < boxesList.length && found < 0; head++) {
    const boxes = boxesList[head]!;
    const player = playerList[head]!;
    for (const b of boxes) occupied[b] = 1;
    region.flood(w, wall, occupied, player);
    const regionStamp = region.stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy, , pushLetter] of DIRS) {
        const d = dy * w + dx;
        const stand = b - d;
        const target = b + d;
        if (region.seen[stand] !== regionStamp) continue;
        if (wall[target] || occupied[target] || !live[target]) continue;
        occupied[b] = 0;
        occupied[target] = 1;
        const frozen = frozenBlock(level, occupied, goal, target);
        let key = -1;
        let nextBoxes: number[] | null = null;
        let walk = '';
        if (!frozen) {
          nextBoxes = boxes.slice();
          nextBoxes[bi] = target;
          nextBoxes.sort((x, y) => x - y);
          // The player ends the push standing on the box's old square.
          key = stateKey(nextBoxes, keyReach.flood(w, wall, occupied, b), size);
          if (!visited.has(key)) {
            occupied[target] = 0;
            occupied[b] = 1;
            walk = walkPath(level, occupied, player, stand) ?? '';
            occupied[b] = 0;
            occupied[target] = 1;
          }
        }
        occupied[target] = 0;
        occupied[b] = 1;
        if (frozen || visited.has(key)) continue;
        if (boxesList.length >= budget) {
          occupied.fill(0);
          return null;
        }
        visited.add(key);
        boxesList.push(nextBoxes!);
        playerList.push(b);
        parent.push(head);
        segment.push(walk + pushLetter);
        if (isSolved(nextBoxes!)) found = boxesList.length - 1;
      }
    }
    occupied.fill(0);
  }
  if (found < 0) return null;
  const parts: string[] = [];
  let pushes = 0;
  for (let s = found; s > 0; s = parent[s]!) {
    parts.push(segment[s]!);
    pushes++;
  }
  parts.reverse();
  return { pushes, moves: parts.join('') };
}

/** Minimum-cost matching of boxes to goals by Manhattan distance (boxes <= 8 handled by permutation search). */
function matchingEstimate(level: Level, boxes: number[], goals: number[]): number {
  const n = boxes.length;
  if (n === 0) return 0;
  const cost: number[][] = boxes.map((b) => {
    const bx = b % level.w;
    const by = Math.floor(b / level.w);
    return goals.map((g) => Math.abs((g % level.w) - bx) + Math.abs(Math.floor(g / level.w) - by));
  });
  // Permutation DP over subsets: dp[mask] = best cost assigning the first popcount(mask) boxes to the goals in mask.
  const full = 1 << n;
  const dp = new Array<number>(full).fill(Infinity);
  dp[0] = 0;
  for (let mask = 0; mask < full; mask++) {
    const k = popcount(mask);
    if (k >= n) continue;
    for (let g = 0; g < n; g++) {
      if (mask & (1 << g)) continue;
      const v = dp[mask]! + cost[k]![g]!;
      const nm = mask | (1 << g);
      if (v < dp[nm]!) dp[nm] = v;
    }
  }
  return dp[full - 1]!;
}

function popcount(x: number): number {
  let c = 0;
  while (x) { x &= x - 1; c++; }
  return c;
}

/**
 * Push-based A* for one subproblem (fallback / whole-level path).
 * The estimate never overcounts, so a finished run is push-optimal;
 * returns null when the expansion budget runs out.
 */
function solveAstar(sub: SubProblem, startBoxes: number[], startPlayer: number, budget: number): Solved | null {
  const { level, live, goal } = sub;
  const { w, wall, goals } = level;
  const size = wall.length;
  if (startBoxes.some((b) => !live[b])) return null;
  const isSolved = (bs: number[]): boolean => bs.every((b) => goal[b] === 1);
  if (isSolved(startBoxes)) return { pushes: 0, moves: '' };

  const occupied = new Uint8Array(size);
  const region = new Reach(size);
  const keyReach = new Reach(size);
  const boxesList: number[][] = [startBoxes.slice()];
  const playerList: number[] = [startPlayer];
  const parent: number[] = [-1];
  const segment: string[] = [''];
  const gScore: number[] = [0];
  const best = new Map<number, number>();
  for (const b of startBoxes) occupied[b] = 1;
  best.set(stateKey(startBoxes, keyReach.flood(w, wall, occupied, startPlayer), size), 0);
  occupied.fill(0);

  // Binary heap of [f, -g, stateIndex].
  const heap: [number, number, number][] = [[matchingEstimate(level, startBoxes, goals), 0, 0]];
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  let expanded = 0;
  let found = -1;
  while (heap.length > 0 && found < 0) {
    const [, , s] = heapPop();
    const boxes = boxesList[s]!;
    const g = gScore[s]!;
    for (const b of boxes) occupied[b] = 1;
    const low = keyReach.flood(w, wall, occupied, playerList[s]!);
    if (best.get(stateKey(boxes, low, size)) !== g) { occupied.fill(0); continue; }
    if (isSolved(boxes)) { found = s; occupied.fill(0); break; }
    if (++expanded > budget) { occupied.fill(0); return null; }
    region.flood(w, wall, occupied, playerList[s]!);
    const regionStamp = region.stamp;
    const player = playerList[s]!;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy, , pushLetter] of DIRS) {
        const d = dy * w + dx;
        const stand = b - d;
        const target = b + d;
        if (region.seen[stand] !== regionStamp) continue;
        if (wall[target] || occupied[target] || !live[target]) continue;
        occupied[b] = 0;
        occupied[target] = 1;
        const frozen = frozenBlock(level, occupied, goal, target);
        let key = -1;
        let nextBoxes: number[] | null = null;
        if (!frozen) {
          nextBoxes = boxes.slice();
          nextBoxes[bi] = target;
          nextBoxes.sort((x, y) => x - y);
          key = stateKey(nextBoxes, keyReach.flood(w, wall, occupied, b), size);
        }
        occupied[target] = 0;
        occupied[b] = 1;
        if (frozen) continue;
        const seenG = best.get(key);
        if (seenG !== undefined && seenG <= g + 1) continue;
        const walk = walkPath(level, occupied, player, stand);
        if (walk === null) continue;
        best.set(key, g + 1);
        const t = boxesList.length;
        boxesList.push(nextBoxes!);
        playerList.push(b);
        parent.push(s);
        segment.push(walk + pushLetter);
        gScore.push(g + 1);
        heapPush([g + 1 + matchingEstimate(level, nextBoxes!, goals), -(g + 1), t]);
      }
    }
    occupied.fill(0);
  }
  if (found < 0) return null;
  const parts: string[] = [];
  let pushes = 0;
  for (let s = found; s > 0; s = parent[s]!) {
    parts.push(segment[s]!);
    pushes++;
  }
  parts.reverse();
  return { pushes, moves: parts.join('') };
}

interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
}

/**
 * The level's rooms, or null when it does not split cleanly: cut the
 * dead squares of the player's row (the hallway), then each connected
 * part of what remains that holds boxes is a room.
 */
function roomsOf(level: Level, live: Uint8Array): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b] === 1)) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares: number[] = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      const p = squares[i]!;
      for (const [dx, dy] of DIRS) {
        const q = p + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) {
    if (part[g] < 0) return null;
    rooms[part[g]!]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** The square the player stands on after playing `moves` from `from`. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) {
    p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  }
  return p;
}

function solveLevel(text: string): string {
  const level = parseLevel(text);
  const live = liveSquares(level);
  const goalMask = new Uint8Array(level.wall.length);
  for (const g of level.goals) goalMask[g] = 1;
  const rooms = roomsOf(level, live);

  if (rooms === null) {
    const sub: SubProblem = { level, live, goal: goalMask };
    const solved = solveAstar(sub, level.boxes, level.player, WHOLE_ASTAR_BUDGET);
    return solved ? solved.moves : '';
  }

  // Solve room by room. While a room is being solved, every other
  // room's boxes stand as walls: at their start squares before their
  // room is done, on their goals afterwards.
  let player = level.player;
  let moves = '';
  const done = new Array<boolean>(rooms.length).fill(false);
  for (let ri = 0; ri < rooms.length; ri++) {
    const room = rooms[ri]!;
    const wall = level.wall.slice();
    for (let oj = 0; oj < rooms.length; oj++) {
      if (oj === ri) continue;
      const other = rooms[oj]!;
      for (const b of done[oj] ? other.goals : other.boxes) wall[b] = 1;
    }
    const subLevel: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const roomLive = liveSquares(subLevel);
    const roomGoal = new Uint8Array(level.wall.length);
    for (const g of room.goals) roomGoal[g] = 1;
    const sub: SubProblem = { level: subLevel, live: roomLive, goal: roomGoal };
    const solved =
      solveBfs(sub, subLevel.boxes, player, ROOM_BFS_BUDGET) ??
      solveAstar(sub, subLevel.boxes, player, ROOM_ASTAR_BUDGET);
    if (solved === null) return '';
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    done[ri] = true;
  }
  return moves;
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  const moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
