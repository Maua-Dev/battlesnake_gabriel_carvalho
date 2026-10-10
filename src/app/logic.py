
from collections import Counter, deque
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import product
import os
import logging
import sqlite3
import json
from pathlib import Path
from time import perf_counter
from .models import GameState, MoveResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DIRECTIONS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}

LOSS, WIN = -1_000_000, 1_000_000

BOT_VERSION = "3.5-maua-logic"





@dataclass(frozen=True)

class Snake:

    id: str

    body: tuple

    health: int





@dataclass(frozen=True)

class State:

    snakes: tuple

    food: frozenset





class SearchTimeout(Exception):

    pass





class OpponentMemory:
    """Persistent, modest behavioral priors. No model training libraries required.

    Opponent names are identities across games, not Battlesnake's per-game IDs.
    These are aggregate tendencies, never guaranteed future moves.
    """

    def __init__(self):
        self.path = os.environ.get("OPPONENT_DB", "/tmp/hiperdriver_opponents.sqlite3")

    def connect(self):
        con = sqlite3.connect(self.path, timeout=0.15)
        con.execute("PRAGMA busy_timeout=150")
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("""CREATE TABLE IF NOT EXISTS snapshots (
            game TEXT NOT NULL, turn INTEGER NOT NULL, snake TEXT NOT NULL,
            name TEXT NOT NULL, x INTEGER NOT NULL, y INTEGER NOT NULL,
            width INTEGER NOT NULL, height INTEGER NOT NULL, wrapped INTEGER NOT NULL,
            food TEXT NOT NULL, PRIMARY KEY(game,turn,snake))""")
        con.execute("""CREATE TABLE IF NOT EXISTS patterns (
            name TEXT NOT NULL, context TEXT NOT NULL, direction TEXT NOT NULL,
            n INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(name,context,direction))""")
        return con

    @staticmethod
    def direction(a, b, width, height, wrapped):
        for direction, (dx, dy) in DIRECTIONS.items():
            x, y = a[0] + dx, a[1] + dy
            if wrapped:
                x, y = x % width, y % height
            if (x, y) == b:
                return direction
        return None

    @staticmethod
    def context(head, food, width, height, wrapped):
        if not wrapped and (head[0] in (0, width - 1) or head[1] in (0, height - 1)):
            return "edge"
        if food:
            dist = min(abs(head[0]-x)+abs(head[1]-y) for x, y in food)
            if dist <= 3:
                return "food_near"
        return "open"

    def observe(self, data, wrapped):
        game = str(data.get("game", {}).get("id", ""))
        if not game:
            return
        turn = int(data.get("turn", 0))
        board = data["board"]
        width, height = board["width"], board["height"]
        food = json.dumps([(f["x"],f["y"]) for f in board.get("food", [])], separators=(",", ":"))
        mine = data["you"]["id"]
        with self.connect() as con:
            for snake in board["snakes"]:
                sid = snake["id"]
                if sid == mine:
                    continue
                # Only count an observed move once; skip gaps in observed turns.
                old = con.execute("""SELECT name,x,y,width,height,wrapped,food FROM snapshots
                    WHERE game=? AND turn=? AND snake=?""", (game,turn-1,sid)).fetchone()
                head = snake["head"]
                name = str(snake.get("name") or sid)[:160]
                inserted = con.execute("""INSERT OR IGNORE INTO snapshots
                    (game,turn,snake,name,x,y,width,height,wrapped,food) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (game,turn,sid,name,head["x"],head["y"],width,height,int(wrapped),food)).rowcount
                if old and inserted and old[0] == name:
                    direction = self.direction((old[1],old[2]),(head["x"],head["y"]),old[3],old[4],bool(old[5]))
                    if direction:
                        old_food = json.loads(old[6])
                        context = self.context((old[1],old[2]),old_food,old[3],old[4],bool(old[5]))
                        for c in ("all",context):
                            con.execute("""INSERT INTO patterns(name,context,direction,n) VALUES (?,?,?,1)
                                ON CONFLICT(name,context,direction) DO UPDATE SET n=n+1""",
                                (name,c,direction))
            # Bound disk growth; aggregate patterns are retained.
            if turn % 20 == 0:
                con.execute("DELETE FROM snapshots WHERE rowid IN (SELECT rowid FROM snapshots WHERE game != ? ORDER BY rowid LIMIT 500)", (game,))

    def priors(self, data, wrapped):
        self.observe(data, wrapped)
        names = {s["id"]: str(s.get("name") or s["id"])[:160]
                 for s in data["board"]["snakes"] if s["id"] != data["you"]["id"]}
        output = {}
        with self.connect() as con:
            for snake in data["board"]["snakes"]:
                if snake["id"] not in names:
                    continue
                context = self.context((snake["head"]["x"], snake["head"]["y"]),
                    [(f["x"],f["y"]) for f in data["board"].get("food", [])],
                    data["board"]["width"],data["board"]["height"],wrapped)
                rows = con.execute("SELECT context,direction,n FROM patterns WHERE name=? AND context IN (?, 'all')",(names[snake["id"]],context)).fetchall()
                contextual = {d:n for c,d,n in rows if c == context}
                global_counts = {d:n for c,d,n in rows if c == 'all'}
                # Blend context and general behavior. Laplace smoothing avoids certainty.
                counts = {d:1 + contextual.get(d,0) + 0.2*global_counts.get(d,0) for d in DIRECTIONS}
                total = sum(counts.values())
                output[snake["id"]] = {d:counts[d]/total for d in DIRECTIONS}
        return output


class Brain:

    def __init__(self, data, budget_ms=None, opponent_priors=None):

        started = perf_counter()

        board = data["board"]

        game = data.get("game", {})

        rules = game.get("ruleset", {})

        self.width, self.height = board["width"], board["height"]

        self.wrapped = rules.get("name") == "wrapped"

        self.damage = max(0, rules.get("settings", {}).get("hazardDamagePerTurn", 0))

        self.hazards = Counter((p["x"], p["y"]) for p in board.get("hazards", []))

        self.me = data["you"]["id"]
        self.opponent_priors = opponent_priors or {}

        self.state = State(tuple(Snake(s["id"], tuple((p["x"], p["y"])

                           for p in s["body"]), s["health"])

                           for s in board["snakes"]),

                           frozenset((p["x"], p["y"]) for p in board["food"]))

        self.competitive = len(self.state.snakes) > 1

        # Reserva tempo para Flask, rede e variacao de CPU. Nao e garantia de SLA.

        timeout = max(1, int(game.get("timeout", 500)))

        configured = max(0, float(os.environ.get("MOVE_BUDGET_MS", "45")))

        requested = configured if budget_ms is None else max(0, budget_ms)

        self.deadline = started + min(requested, timeout * 0.24) / 1000
        self.started = started

        self.neighbors = {}

        for x in range(self.width):

            for y in range(self.height):

                self.neighbors[x, y] = tuple(self.step((x, y), m) for m in DIRECTIONS

                                             if self.inside(self.step((x, y), m)))

        self.nodes = 0

        self.completed_depth = 0

        self.leaf_cache = {}

        self.search_cache = {}

        self.move_hints = {}

        self.response_hints = {}
        self.region_cache = {}



    def check_time(self):

        if perf_counter() >= self.deadline:

            raise SearchTimeout



    def step(self, point, move):

        dx, dy = DIRECTIONS[move]

        x, y = point[0] + dx, point[1] + dy

        return (x % self.width, y % self.height) if self.wrapped else (x, y)



    def inside(self, p):

        return 0 <= p[0] < self.width and 0 <= p[1] < self.height



    def health_after(self, snake, point, food):
        # Turn order: health decreases, food restores, then hazard damage.
        health = 100 if point in food else snake.health - 1
        return health - self.damage * self.hazards[point]

    @staticmethod
    def occupied(state):
        # A repeated tail coordinate means that cell stays occupied after moving.
        return {p for s in state.snakes for p in s.body[:-1]}

    def moves(self, state, snake):
        blocked = self.occupied(state)
        return [m for m in DIRECTIONS
                if self.inside(p := self.step(snake.body[0], m))
                and p not in blocked
                and self.health_after(snake, p, state.food) > 0]

    def advance(self, state, choices):
        """Resolve simultaneous turns, including growth and vacating tails."""
        moved, eaten = [], set()
        for snake in state.snakes:
            head = self.step(snake.body[0], choices[snake.id])
            feeding = head in state.food
            # Eating retains the OLD tail; otherwise the old tail vacates.
            body = (head,) + (snake.body if feeding else snake.body[:-1])
            health = self.health_after(snake, head, state.food)
            if feeding:
                eaten.add(head)
            if health > 0 and self.inside(head):
                moved.append(Snake(snake.id, body, health))

        survivors = []
        for snake in moved:
            head = snake.body[0]
            if any(head in other.body[1:] for other in moved):
                continue
            if any(other.id != snake.id and other.body[0] == head
                   and len(other.body) >= len(snake.body) for other in moved):
                continue
            survivors.append(snake)
        return State(tuple(survivors), state.food - eaten)

    def distances(self, start, occupied):

        distance, queue = {start: 0}, deque([start])

        while queue:

            p = queue.popleft()

            for q in self.neighbors.get(p, ()):

                if q not in occupied and q not in distance:

                    distance[q] = distance[p] + 1

                    queue.append(q)

        return distance



    def food_cost(self, snake, state, occupied):

        """Menor custo de saude ate comida, respeitando corpos e hazards.



        A ultima casa pode ser alcancada com 1 de saude e restaura a saude.

        Corpo estatico: estimativa conservadora; a busca simula caudas depois.

        """

        start = snake.body[0]

        costs, heap = {start: 0}, [(0, start)]

        while heap:

            cost, p = heappop(heap)

            if cost != costs[p]:

                continue

            for q in self.neighbors.get(p, ()):

                if q in occupied:

                    continue

                if q in state.food:

                    return cost + 1

                new_cost = cost + 1 + self.damage * self.hazards[q]

                if new_cost < costs.get(q, float("inf")):

                    costs[q] = new_cost

                    heappush(heap, (new_cost, q))

        return float("inf")



    def safe_region(self, state, own, occupied):
        cached = self.region_cache.get(state)
        if cached is not None:
            return cached
        enemies = [s for s in state.snakes if s.id != self.me]
        my_dist = self.distances(own.body[0], occupied)
        enemy_dist = [(s, self.distances(s.body[0], occupied)) for s in enemies]
        count = 0
        for point, distance in my_dist.items():
            if all(distance < d.get(point, 10_000) or
                   (distance == d.get(point, 10_000) and len(own.body) > len(enemy.body))
                   for enemy, d in enemy_dist):
                count += 1
        self.region_cache[state] = count
        return count

    def evaluate(self, state):

        cached = self.leaf_cache.get(state)

        if cached is not None:

            return cached

        own = next(s for s in state.snakes if s.id == self.me)

        enemies = [s for s in state.snakes if s.id != self.me]

        occupied = self.occupied(state)

        maps = {s.id: self.distances(s.body[0], occupied) for s in state.snakes}

        mine = maps[self.me]

        size = len(own.body)

        space = len(mine)
        # Prioritize escape routes over a deceptive open area connected by one bottleneck.
        secure_space = self.safe_region(state, own, occupied)

        # Nao confundir bonus de centro com quantidade de casas disponiveis.

        score = 2.0 * min(secure_space, size * 3) - 115 * max(0, size + 2 - secure_space)

        territory = 0

        for point, distance in mine.items():

            if all(maps[e.id].get(point, 10_000) > distance or

                   (maps[e.id].get(point) == distance and size > len(e.body))

                   for e in enemies):

                territory += 1

        score += 3.5 * territory

        score -= 48 * max(0, size + 2 - territory)

        exits = len(self.moves(state, own))

        score += 12 * exits

        if exits == 0:

            score -= 15_000

        elif exits == 1:

            score -= 75

        # Acesso a cauda e um sinal auxiliar, nao uma prova de seguranca.

        if own.body[-1] in mine:

            score += 20

        # A arena repete a direcao anterior quando a resposta atrasa. Entre

        # posicoes semelhantes, prefira deixar uma casa livre a frente.

        # Penalidade limitada: nao pode superar uma morte ou vitoria comprovada.

        if len(own.body) > 1 and own.body[0] != own.body[1]:

            for direction in DIRECTIONS:

                if self.step(own.body[1], direction) == own.body[0]:

                    forward = self.step(own.body[0], direction)

                    if not self.inside(forward) or forward in occupied:

                        score -= 60

                    break

        score += 1.2 * own.health

        enemy_size = max((len(s.body) for s in enemies), default=size)

        score += 35 * min(size, enemy_size + 2)

        if not self.damage:

            # Sem hazards, o BFS ja calculado e o custo exato no mapa estatico.

            # Evita repetir uma busca Dijkstra em cada folha do minimax.

            cost = min((mine[f] for f in state.food if f in mine and f != own.body[0]),

                       default=float("inf"))

        else:

            cost = self.food_cost(own, state, occupied) if state.food else float("inf")

        if cost <= own.health:

            urgency = 8 + max(0, 55 - own.health) * 1.5

            score += urgency * 12 / (cost + 1)

            score -= max(0, cost + 6 - own.health) * 65

        else:

            score -= max(0, 45 - own.health) * 35

        # Pressao sobre o territorio inimigo, sem perseguir a cabeca cegamente.

        for enemy in enemies:

            if size > len(enemy.body) and own.health > 30 and space >= size + 3:

                enemy_exits = len(self.moves(state, enemy))

                score += 24 * (3 - enemy_exits)

        if not self.wrapped:

            x, y = own.body[0]

            score -= 0.4 * (abs(x - (self.width - 1) / 2) + abs(y - (self.height - 1) / 2))

        if self.damage:

            score -= self.hazards[own.body[0]] * 18

        self.leaf_cache[state] = score

        return score



    def immediate_rank(self, state, own, move):

        """Fallback barato calculado antes da busca; ataques sao subordinados a risco."""

        p = self.step(own.body[0], move)

        occupied = self.occupied(state)

        health = self.health_after(own, p, state.food)

        if not self.inside(p) or p in occupied or health <= 0:

            return LOSS

        body = (p,) + (own.body if p in state.food else own.body[:-1])

        projected = Snake(own.id, body, health)

        # Heuristica apenas: posicoes inimigas ainda sao do turno atual.

        projected_state = State(tuple(projected if s.id == own.id else s for s in state.snakes),

                                state.food - {p})

        risk, opportunity, likelihood = 0, 0, 0.0

        for enemy in state.snakes:

            if enemy.id == own.id:

                continue

            for enemy_move in self.moves(state, enemy):

                if self.step(enemy.body[0], enemy_move) == p:

                    enemy_length = len(enemy.body) + int(p in state.food)

                    if enemy_length >= len(body):

                        risk += 1
                        likelihood += self.opponent_priors.get(enemy.id, {}).get(enemy_move, 0.25)

                    else:

                        opportunity += 1

        score = self.evaluate(projected_state) - 250_000 * risk - 250 * likelihood + 30 * opportunity
        # Avoid closing into a pocket with fewer squares than body length.
        reachable = self.distances(p, self.occupied(projected_state))
        if len(reachable) < len(body) + 1:
            score -= 4000 + 200 * (len(body) + 1 - len(reachable))
        return score



    def escape_plan(self, state, own, initial_move, horizon=None, beam_width=24, time_limit=None):
        """Beam search of the moving body, with pessimistic contested-territory scoring.

        Enemy bodies are kept stationary here (Minimax models their real moves).
        Reachable squares alone are not evidence of an exit: penalize bottlenecks,
        inaccessible tail and squares enemies can contest before us.
        Returns (survived_depth, minimum_safe_area, trajectory_quality).
        """
        if horizon is None:
            horizon = min(13, max(8, len(own.body) + 2))
        if time_limit is None:
            time_limit = self.deadline
        enemies = [s for s in state.snakes if s.id != own.id]
        enemy_blocked = {p for s in enemies for p in s.body}
        static_occupied = self.occupied(state)
        enemy_maps = [(s, self.distances(s.body[0], static_occupied)) for s in enemies]

        def next_steps(body, health, food):
            blocked = set(body[:-1]) | enemy_blocked
            for m in DIRECTIONS:
                p = self.step(body[0], m)
                if not self.inside(p) or p in blocked:
                    continue
                feeding = p in food
                hp = (100 if feeding else health - 1) - self.damage * self.hazards[p]
                if hp > 0:
                    yield m, (p,) + (body if feeding else body[:-1]), hp, food - {p}

        first = next((v for v in next_steps(own.body, own.health, state.food)
                      if v[0] == initial_move), None)
        if first is None:
            return (0, 0, -100000)
        _, body, hp, food = first
        # depth, minimum viable area, and a bounded quality reward
        layer = [(body, hp, food, self.width * self.height, 0.0)]
        best = (0, 0, -100000)
        for depth in range(1, horizon + 1):
            if perf_counter() >= time_limit:
                break
            scored = []
            for body, hp, food, minimum, total in layer:
                blocked = set(body[:-1]) | enemy_blocked
                distances = self.distances(body[0], blocked)
                area = len(distances)
                # Competitor may reach shared land sooner. This is a heuristic,
                # not a proof that any particular square becomes blocked.
                safe_area = sum(1 for point, dist in distances.items()
                                if all(dist < emap.get(point, 10000) or
                                       (dist == emap.get(point, 10000) and
                                        len(body) > len(enemy.body))
                                       for enemy, emap in enemy_maps))
                exits = sum(1 for q in self.neighbors[body[0]] if q not in blocked)
                tail_visible = body[-1] in distances
                margin = safe_area - len(body)
                # Penalize thin territories even if a long self-following route
                # would make it technically possible to survive the horizon.
                local = 2.0 * min(25, margin) + 11 * exits
                local += 16 if tail_visible else -25
                local -= 85 * max(0, 3 - margin)
                if exits <= 1:
                    local -= 45
                if area > safe_area:
                    local -= min(65, 1.5 * (area - safe_area))
                new_minimum = min(minimum, safe_area)
                new_total = total + local
                candidate = (depth, new_minimum, new_total / depth)
                if candidate > best:
                    best = candidate
                if depth < horizon:
                    for _, b, h, f in next_steps(body, hp, food):
                        scored.append((b, h, f, new_minimum, new_total))
            if not scored:
                break
            unique = {}
            for item in scored:
                key = item[:3]
                if key not in unique or item[3:] > unique[key][3:]:
                    unique[key] = item
            layer = sorted(unique.values(), key=lambda x: (x[3], x[4], x[1]),
                           reverse=True)[:beam_width]
        return best

    def root_threat(self, state, own, move):
        """Immediate, adversarial head-collision risk; no guess from learned priors."""
        point = self.step(own.body[0], move)
        if not self.inside(point) or point in self.occupied(state) or \
                self.health_after(own, point, state.food) <= 0:
            return True
        our_length = len(own.body) + int(point in state.food)
        for enemy in state.snakes:
            if enemy.id == own.id:
                continue
            for enemy_move in self.moves(state, enemy):
                if self.step(enemy.body[0], enemy_move) == point:
                    their_length = len(enemy.body) + int(point in state.food)
                    if their_length >= our_length:
                        return True
        return False

    def terminal(self, state, ply):

        if not any(s.id == self.me for s in state.snakes):

            return LOSS + 100 * ply

        if self.competitive and len(state.snakes) == 1:

            return WIN - 100 * ply

        return None



    def search(self, state, depth, ply=0, preferred=None):

        self.check_time()

        self.nodes += 1

        result = self.terminal(state, ply)

        if result is not None:

            return result, None

        if depth == 0:

            return self.evaluate(state), None

        key = (state, depth, ply)

        if key in self.search_cache:

            return self.search_cache[key]

        own = next(s for s in state.snakes if s.id == self.me)

        options = self.moves(state, own) or list(DIRECTIONS)
        # Na raiz, uma colisao fatal de cabeca nunca e compensada por heuristica.
        # Fora da raiz mantemos todas as alternativas para modelar combate.
        if ply == 0 and hasattr(self, "root_candidates"):
            options = [m for m in options if m in self.root_candidates] or options

        enemies = [s for s in state.snakes if s.id != self.me]

        # Oponentes sem saidas ainda se movem e podem causar colisoes antes de morrer.

        enemy_options = [self.moves(state, s) or list(DIRECTIONS) for s in enemies]
        # Learned priors only reorder responses; all legal opponent moves remain
        # considered, so worst-case safety is not replaced with blind prediction.
        if ply == 0:
            enemy_options = [sorted(opts, key=lambda m: -self.opponent_priors.get(s.id, {}).get(m, 0.25))
                             for s, opts in zip(enemies, enemy_options)]

        if preferred is None:

            preferred = self.move_hints.get(state)

        if preferred in options:

            options.remove(preferred)

            options.insert(0, preferred)

        best_score, best_move = -float("inf"), options[0]

        for move in options:

            worst = float("inf")

            # Testar primeiro a refutacao encontrada na iteracao anterior.

            # Todas as respostas continuam disponiveis; muda apenas a ordem.

            hint_key = (state, move)

            response_hint = self.response_hints.get(hint_key)

            responses = list(product(*enemy_options))

            if response_hint in responses:

                responses.remove(response_hint)

                responses.insert(0, response_hint)

            worst_response = None

            for response in responses:

                self.check_time()

                choices = {s.id: m for s, m in zip(enemies, response)}

                choices[own.id] = move

                child = self.advance(state, choices)

                value, _ = self.search(child, depth - 1, ply + 1)

                if value < worst:

                    worst, worst_response = value, response

                # Nenhuma resposta restante pode melhorar o pior caso deste movimento.

                if worst <= best_score:

                    break

            if worst_response is not None:

                self.response_hints[hint_key] = worst_response

            # Penalizar precoce encurralamento sem superar morte/vitoria certa.
            # Escape incompleto (prazo de CPU) recebe peso reduzido; a busca
            # minimax ainda e a principal avaliacao tatica do movimento.
            if ply == 0 and hasattr(self, "root_escape") and abs(worst) < 900_000:
                esc = self.root_escape.get(move)
                if esc:
                    depth_survived, minimum_area, quality = esc
                    size = len(own.body)
                    room_deficit = max(0, size + 2 - minimum_area)
                    early_trap = max(0, min(7, size + 2) - depth_survived)
                    adjustment = (10 * min(depth_survived, 13)
                                  + 3 * min(minimum_area, 50)
                                  + 0.4 * max(-600, min(600, quality))
                                  - 155 * room_deficit - 225 * early_trap)
                    worst += max(-3500, min(650, adjustment))
            if worst > best_score:

                best_score, best_move = worst, move

        self.search_cache[key] = (best_score, best_move)

        self.move_hints[state] = best_move

        return best_score, best_move



    def choose(self):
        own = next(s for s in self.state.snakes if s.id == self.me)
        options = self.moves(self.state, own) or list(DIRECTIONS)
        if len(options) == 1:
            return options[0]

        # A tactical risk is never made safe by a favorable escape heuristic.
        unthreatened = [m for m in options if not self.root_threat(self.state, own, m)]
        candidates = unthreatened or options
        self.root_candidates = tuple(candidates)
        selected = max(candidates, key=lambda m: self.immediate_rank(self.state, own, m))

        escape = {}
        escape_deadline = self.started + (self.deadline - self.started) * 0.43
        for m in candidates:
            if perf_counter() >= escape_deadline:
                break
            escape[m] = self.escape_plan(self.state, own, m, time_limit=escape_deadline)
        self.root_escape = escape
        if escape:
            # Prioritize preservation of viable territory, not just horizon.
            selected = max(escape, key=lambda m: (
                escape[m][0], escape[m][1], escape[m][2],
                self.immediate_rank(self.state, own, m)))

        # Search remains authoritative for forced wins / proven tactical deaths.
        # Completed iterations only; timeout cannot replace a complete answer.
        for depth in range(1, 9):
            try:
                score, move = self.search(self.state, depth, preferred=selected)
            except SearchTimeout:
                break
            if move is not None:
                selected = move
            self.completed_depth = depth
            if score >= WIN - 10_000 or score <= LOSS + 10_000:
                break
        return selected



def choose_move(data, budget_ms=None):
    """Pure strategy call; persistent learning is performed by the /move route."""
    return Brain(data, budget_ms).choose()





def _value(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _point(point):
    return {"x": int(_value(point, "x")), "y": int(_value(point, "y"))}


def _to_payload(state):
    """Converte os modelos do template Maua Dev para o formato do Brain."""
    board = _value(state, "board")
    you = _value(state, "you")
    game = _value(state, "game")
    ruleset = _value(game, "ruleset")
    settings = _value(ruleset, "settings")
    def snake_payload(s):
        body = [_point(p) for p in _value(s, "body", [])]
        head = _point(_value(s, "head", body[0] if body else None))
        return {"id": str(_value(s, "id")), "name": str(_value(s, "name", "") or ""),
                "health": int(_value(s, "health", 100)), "body": body,
                "head": head, "length": len(body)}
    snakes = [snake_payload(s) for s in _value(board, "snakes", [])]
    mine = snake_payload(you)
    if not any(s["id"] == mine["id"] for s in snakes):
        snakes.append(mine)
    return {
        "game": {"id": str(_value(game, "id", "")),
                 "timeout": int(_value(game, "timeout", 500) or 500),
                 "ruleset": {"name": str(_value(ruleset, "name", "standard")),
                             "settings": {"hazardDamagePerTurn": int(_value(settings, "hazardDamagePerTurn", 0) or 0)}}},
        "turn": int(_value(state, "turn", 0)),
        "board": {"width": int(_value(board, "width")), "height": int(_value(board, "height")),
                  "snakes": snakes, "food": [_point(p) for p in _value(board, "food", [])],
                  "hazards": [_point(p) for p in _value(board, "hazards", [])]},
        "you": mine,
    }


def info() -> dict:
    return {"apiversion": "1", "author": "Gabriel", "color": "#0000FF",
            "head": "all-seeing", "tail": "bolt", "version": BOT_VERSION}


def start(state: GameState) -> None:
    logger.info("Hiperdriver iniciou partida %s", _value(_value(state, "game"), "id"))


def end(state: GameState) -> None:
    logger.info("Hiperdriver encerrou partida no turno %s", _value(state, "turn"))


def get_move(state: GameState) -> MoveResponse:
    started = perf_counter()
    try:
        data = _to_payload(state)
        priors = {}
        # Em Lambda /tmp pode ser apagado a qualquer momento; nunca depende dele.
        if os.environ.get("HIPERDRIVER_MEMORY", "0") == "1":
            try:
                priors = OpponentMemory().priors(
                    data, data["game"]["ruleset"]["name"] == "wrapped")
            except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as exc:
                logger.warning("Memoria indisponivel: %s", exc)
        brain = Brain(data, opponent_priors=priors)
        direction = brain.choose()
        logger.info("Hiperdriver turno=%s movimento=%s profundidade=%s ms=%.1f",
                    data["turn"], direction, brain.completed_depth,
                    (perf_counter()-started)*1000)
        return MoveResponse(move=direction)
    except Exception:
        logger.exception("Falha na estrategia da Hiperdriver; aplicando contingencia")
        # Preservar uma resposta valida mesmo em falhas inesperadas.
        head = _value(_value(state, "you"), "head")
        if head is None:
            head = _value(_value(state, "you"), "body")[0]
        board = _value(state, "board")
        occupied = {(_value(p, "x"), _value(p, "y"))
                    for s in _value(board, "snakes", []) for p in list(_value(s, "body", []))[:-1]}
        x, y = _value(head, "x"), _value(head, "y")
        for direction, (dx, dy) in DIRECTIONS.items():
            nx, ny = x+dx, y+dy
            if (0 <= nx < _value(board, "width") and 0 <= ny < _value(board, "height")
                    and (nx,ny) not in occupied):
                return MoveResponse(move=direction)
        return MoveResponse(move="up")
