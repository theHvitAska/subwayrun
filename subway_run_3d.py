#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D 地铁跑酷  (Subway Surfers 风格)
================================
纯 Python + Pygame 手写软件 3D 渲染，不依赖 OpenGL / 3D 引擎。

渲染管线：
    世界坐标 -> 相机空间(点积投影) -> 近平面裁剪(Sutherland-Hodgman)
             -> 透视投影 -> 背面剔除 -> 画家算法排序 -> 距离雾 -> 多边形填充

操作：
    ← → / A D    左右换道
    ↑ / W / 空格 跳跃
    ↓ / S        翻滚（空中按 = 快速下坠）
    P            暂停
    R            重新开始
    ESC          退出
"""

import json
import math
import os
import random
import sys

import pygame

# ---------------------------------------------------------------------------
# 基础常量
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 1280, 720
FPS = 60

NEAR = 0.25                       # 近裁剪面
FOV_Y = math.radians(62.0)        # 垂直视角
FOCAL = (HEIGHT * 0.5) / math.tan(FOV_Y * 0.5)

FOG_START, FOG_END = 28.0, 110.0  # 雾的起止距离
FOG_COLOR = (37, 50, 80)

SKY_TOP = (10, 13, 24)
SKY_BOT = (26, 32, 48)
HORIZON_GLOW = (58, 82, 132)

LIGHT = (-0.35, 0.85, -0.40)      # 主光：左上前方
_ln = math.sqrt(sum(c * c for c in LIGHT))
LIGHT = tuple(c / _ln for c in LIGHT)
LIGHT2 = (0.50, 0.20, 0.30)       # 补光：右侧，避免侧墙全黑
_ln2 = math.sqrt(sum(c * c for c in LIGHT2))
LIGHT2 = tuple(c / _ln2 for c in LIGHT2)

LANES = (-2.35, 0.0, 2.35)        # 三条跑道
TRACK_HALF = 4.55                 # 路面半宽
SPAWN_Z = 118.0                   # 障碍物生成位置（雾里）
DESPAWN_Z = -20.0

GRAVITY = 42.0
JUMP_V = 14.2
LANE_SPEED = 17.0                 # 换道插值速度
ROLL_TIME = 0.55

SPEED_BASE, SPEED_MAX = 17.0, 37.0

PLAYER_H = 1.86                   # 站立碰撞高度
ROLL_H = 0.86                     # 翻滚碰撞高度

# 调色板
C_HOODIE = (54, 132, 232)
C_JEANS = (46, 58, 92)
C_SKIN = (246, 206, 168)
C_CAP = (228, 72, 72)
C_PACK = (246, 190, 62)
C_SHOE = (232, 232, 240)

GROUND_A = (49, 55, 71)
GROUND_B = (45, 51, 67)
C_LINE = (146, 162, 198)
WALL_A = (74, 82, 108)
WALL_B = (64, 71, 95)
C_NEON = ((86, 224, 240), (240, 112, 192))
C_GOLD = (255, 198, 60)
C_GOLD2 = (255, 232, 150)

TRAIN_COLORS = ((198, 66, 74), (58, 132, 198), (232, 148, 52), (86, 168, 128))


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def clamp(v, a, b):
    return a if v < a else (b if v > b else v)


def mix(c1, c2, t):
    t = clamp(t, 0.0, 1.0)
    return (int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t))


def clip_near(poly):
    """把相机空间的凸多边形按 z >= NEAR 裁剪。"""
    out = []
    n = len(poly)
    prev = poly[-1]
    prev_in = prev[2] >= NEAR
    for i in range(n):
        cur = poly[i]
        cur_in = cur[2] >= NEAR
        if cur_in != prev_in:
            t = (NEAR - prev[2]) / (cur[2] - prev[2])
            out.append((prev[0] + (cur[0] - prev[0]) * t,
                        prev[1] + (cur[1] - prev[1]) * t,
                        NEAR))
        if cur_in:
            out.append(cur)
        prev = cur
        prev_in = cur_in
    return out


# 立方体 8 个顶点（相对中心），以及 6 个面的顶点索引 + 外法线
_BOX_LOCAL = ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
              (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))
_BOX_FACES = (
    ((4, 5, 6, 7), (0, 0, 1)),     # 前
    ((1, 0, 3, 2), (0, 0, -1)),    # 后
    ((0, 4, 7, 3), (-1, 0, 0)),    # 左
    ((5, 1, 2, 6), (1, 0, 0)),     # 右
    ((3, 7, 6, 2), (0, 1, 0)),     # 上
    ((0, 1, 5, 4), (0, -1, 0)),    # 下
)


# ---------------------------------------------------------------------------
# 相机
# ---------------------------------------------------------------------------
class Camera:
    __slots__ = ("pos", "right", "up", "fwd")

    def __init__(self):
        self.pos = (0.0, 3.0, -8.0)
        self.right = (1.0, 0.0, 0.0)
        self.up = (0.0, 1.0, 0.0)
        self.fwd = (0.0, 0.0, 1.0)

    def look_at(self, pos, target, roll=0.0):
        self.pos = pos
        fx = target[0] - pos[0]
        fy = target[1] - pos[1]
        fz = target[2] - pos[2]
        d = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
        fx, fy, fz = fx / d, fy / d, fz / d

        # right = normalize(cross(worldUp, fwd)) ，worldUp = (0,1,0)
        rx, ry, rz = fz, 0.0, -fx
        d = math.sqrt(rx * rx + rz * rz) or 1.0
        rx, rz = rx / d, rz / d

        # up = cross(fwd, right)
        ux = fy * rz - fz * ry
        uy = fz * rx - fx * rz
        uz = fx * ry - fy * rx

        if roll:
            cr, sr = math.cos(roll), math.sin(roll)
            rx, ux = rx * cr + ux * sr, ux * cr - rx * sr
            ry, uy = ry * cr + uy * sr, uy * cr - ry * sr
            rz, uz = rz * cr + uz * sr, uz * cr - rz * sr

        self.fwd = (fx, fy, fz)
        self.right = (rx, ry, rz)
        self.up = (ux, uy, uz)

    def to_cam(self, p):
        dx = p[0] - self.pos[0]
        dy = p[1] - self.pos[1]
        dz = p[2] - self.pos[2]
        r = self.right
        u = self.up
        f = self.fwd
        return (dx * r[0] + dy * r[1] + dz * r[2],
                dx * u[0] + dy * u[1] + dz * u[2],
                dx * f[0] + dy * f[1] + dz * f[2])


# ---------------------------------------------------------------------------
# 渲染器：收集面片 -> 排序 -> 填充
# ---------------------------------------------------------------------------
class Renderer:
    def __init__(self, surface):
        self.surf = surface
        self.cam = Camera()
        self.faces = []

    def begin(self):
        self.faces.clear()

    # ---- 核心：一个面片入队 ------------------------------------------------
    def push(self, cverts, normal, vis_pt, color, bias=0.0, lit=True):
        # 背面剔除
        if normal is not None:
            px, py, pz = self.cam.pos
            if ((vis_pt[0] - px) * normal[0] +
                    (vis_pt[1] - py) * normal[1] +
                    (vis_pt[2] - pz) * normal[2]) >= 0.0:
                return

        mx = mn = cverts[0][2]
        for c in cverts:
            z = c[2]
            if z > mx:
                mx = z
            elif z < mn:
                mn = z
        if mx < NEAR:
            return

        poly = cverts if mn >= NEAR else clip_near(cverts)
        if len(poly) < 3:
            return

        depth = 0.0
        for p in poly:
            depth += p[2]
        depth /= len(poly)
        if depth > FOG_END + 16.0:
            return

        pts = [(WIDTH * 0.5 + FOCAL * p[0] / p[2],
                HEIGHT * 0.5 - FOCAL * p[1] / p[2]) for p in poly]

        c = color
        if lit and normal is not None:
            lam = (normal[0] * LIGHT[0] + normal[1] * LIGHT[1] +
                   normal[2] * LIGHT[2])
            lam2 = (normal[0] * LIGHT2[0] + normal[1] * LIGHT2[1] +
                    normal[2] * LIGHT2[2])
            b = (0.58 + 0.34 * (lam if lam > 0.0 else 0.0) +
                 0.18 * (lam2 if lam2 > 0.0 else 0.0))
            c = (min(255, int(c[0] * b)),
                 min(255, int(c[1] * b)),
                 min(255, int(c[2] * b)))

        if depth > FOG_START:
            f = (depth - FOG_START) / (FOG_END - FOG_START)
            if f > 1.0:
                f = 1.0
            c = mix(c, FOG_COLOR, f * 0.96)

        self.faces.append((depth + bias, pts, c))

    # ---- 四边形（世界坐标） -----------------------------------------------
    def quad(self, pts, color, normal=None, bias=0.0, lit=True):
        cv = [self.cam.to_cam(p) for p in pts]
        self.push(cv, normal, pts[0] if normal else None, color, bias, lit)

    # ---- 立方体 ------------------------------------------------------------
    def box(self, cx, cy, cz, sx, sy, sz, color, bias=0.0, yaw=0.0):
        hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
        ca = math.cos(yaw) if yaw else 1.0
        sa = math.sin(yaw) if yaw else 0.0
        verts = []
        for ux, uy, uz in _BOX_LOCAL:
            lx, ly, lz = ux * hx, uy * hy, uz * hz
            verts.append((cx + lx * ca + lz * sa,
                          cy + ly,
                          cz - lx * sa + lz * ca))
        cv = [self.cam.to_cam(v) for v in verts]
        for idxs, nrm in _BOX_FACES:
            nx, ny, nz = nrm
            if yaw:
                nx, nz = nx * ca + nz * sa, -nx * sa + nz * ca
            face = (cv[idxs[0]], cv[idxs[1]], cv[idxs[2]], cv[idxs[3]])
            self.push(face, (nx, ny, nz), verts[idxs[0]], color, bias)

    # ---- 始终面向相机的多边形（圆盘 / 金币） -------------------------------
    def disc(self, x, y, z, rx, ry, color, sides=10, bias=0.0, spin=0.0):
        r = self.cam.right
        u = self.cam.up
        step = 6.283185307179586 / sides
        if sides % 2 == 0:
            step = 6.283185307179586 / sides
        pts = []
        for i in range(sides):
            a = i * step + spin
            ca = math.cos(a) * rx
            sa = math.sin(a) * ry
            pts.append((x + r[0] * ca + u[0] * sa,
                        y + r[1] * ca + u[1] * sa,
                        z + r[2] * ca + u[2] * sa))
        cv = [self.cam.to_cam(p) for p in pts]
        self.push(cv, None, None, color, bias, lit=False)

    # ---- 始终面向相机的广告牌 ---------------------------------------------
    def billboard(self, x, y, z, hw, hh, color, bias=0.0):
        r = self.cam.right
        u = self.cam.up
        pts = (
            (x - r[0] * hw - u[0] * hh, y - r[1] * hw - u[1] * hh, z - r[2] * hw - u[2] * hh),
            (x + r[0] * hw - u[0] * hh, y + r[1] * hw - u[1] * hh, z + r[2] * hw - u[2] * hh),
            (x + r[0] * hw + u[0] * hh, y + r[1] * hw + u[1] * hh, z + r[2] * hw + u[2] * hh),
            (x - r[0] * hw + u[0] * hh, y - r[1] * hw + u[1] * hh, z - r[2] * hw + u[2] * hh),
        )
        cv = [self.cam.to_cam(p) for p in pts]
        self.push(cv, None, None, color, bias, lit=False)

    def flush(self):
        self.faces.sort(key=lambda f: f[0], reverse=True)
        draw = pygame.draw.polygon
        surf = self.surf
        for _, pts, col in self.faces:
            draw(surf, col, pts, 0)


# ---------------------------------------------------------------------------
# 字体
# ---------------------------------------------------------------------------
def _find_cjk_font():
    for name in ("microsoftyahei", "microsoft yahei", "msyh", "msyhbd", "simhei",
                 "dengxian", "simsun", "notosanscjksc", "notosanscjk",
                 "sourcehansanssc", "pingfangsc", "heiti", "stheiti"):
        try:
            p = pygame.font.match_font(name)
        except Exception:
            p = None
        if p:
            return p
    return None


CJK_FONT = None
CN = False
_FONT_CACHE = {}


def get_font(size, bold=False):
    key = (size, bold)
    f = _FONT_CACHE.get(key)
    if f is None:
        if CJK_FONT:
            f = pygame.font.Font(CJK_FONT, size)
            f.set_bold(bold)
        else:
            f = pygame.font.SysFont("arial", size, bold=bold)
        _FONT_CACHE[key] = f
    return f


def T(cn_text, en_text):
    return cn_text if CN else en_text


# ---------------------------------------------------------------------------
# 游戏对象
# ---------------------------------------------------------------------------
class Obstacle:
    __slots__ = ("kind", "x", "y0", "z", "sx", "sy", "sz", "color")

    def __init__(self, kind, lane_x, y0, z, sx, sy, sz, color=(200, 80, 80)):
        self.kind = kind
        self.x = lane_x
        self.y0 = y0
        self.z = z
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.color = color


class Coin:
    __slots__ = ("x", "y", "z", "spin")

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.spin = random.uniform(0.0, 6.28)


class Player:
    def __init__(self):
        self.reset()

    def reset(self):
        self.lane = 1
        self.x = 0.0
        self.y = 0.0
        self.vy = 0.0
        self.target_x = 0.0
        self.rolling = 0.0
        self.phase = 0.0
        self.tilt = 0.0
        self.just_landed = False

    @property
    def height(self):
        return ROLL_H if self.rolling > 0.0 else PLAYER_H

    def update(self, dt, speed):
        # 换道：指数平滑，帧率无关
        k = 1.0 - math.exp(-LANE_SPEED * dt)
        self.x += (self.target_x - self.x) * k
        self.tilt = clamp((self.target_x - self.x) * 0.32, -0.5, 0.5)

        # 跳跃
        if self.y > 0.0 or self.vy != 0.0:
            self.vy -= GRAVITY * dt
            self.y += self.vy * dt
            self.just_landed = False
            if self.y <= 0.0:
                if self.vy < -6.0:
                    self.just_landed = True
                self.y = 0.0
                self.vy = 0.0

        if self.rolling > 0.0:
            self.rolling = max(0.0, self.rolling - dt)

        self.phase += dt * (5.0 + speed * 0.32)

    # --- 动作 ---
    def move(self, d):
        self.lane = int(clamp(self.lane + d, 0, 2))
        self.target_x = LANES[self.lane]

    def jump(self):
        if self.y <= 0.001:
            self.vy = JUMP_V
            self.y = 0.02
            self.rolling = 0.0

    def roll(self):
        if self.y > 0.05:
            self.vy = min(self.vy, -JUMP_V * 1.4)   # 空中快速下坠
        self.rolling = ROLL_TIME


# ---------------------------------------------------------------------------
# 主游戏
# ---------------------------------------------------------------------------
class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("3D 地铁跑酷 · Python 复刻")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

        global CJK_FONT, CN
        CJK_FONT = _find_cjk_font()
        CN = CJK_FONT is not None

        self.renderer = Renderer(self.screen)
        self.cam = self.renderer.cam
        self.bg = self._make_background()
        self.overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        self.overlay.fill((6, 9, 16, 205))
        self.flash_surf = pygame.Surface((WIDTH, HEIGHT))
        self.flash_surf.fill((255, 226, 140))

        self.f_score = get_font(44, True)
        self.f_label = get_font(19)
        self.f_mid = get_font(28, True)
        self.f_big = get_font(74, True)
        self.f_small = get_font(22)

        self.best = self._load_best()
        self.player = Player()
        self.reset()
        self.running = True

    # ---------------- 数据持久化 ----------------
    def _best_path(self):
        return os.path.join(os.path.expanduser("~"), ".subway_run_3d.json")

    def _load_best(self):
        try:
            with open(self._best_path(), "r", encoding="utf-8") as f:
                return int(json.load(f).get("best", 0))
        except Exception:
            return 0

    def _save_best(self):
        try:
            with open(self._best_path(), "w", encoding="utf-8") as f:
                json.dump({"best": self.best}, f)
        except Exception:
            pass

    # ---------------- 初始化 / 重开 ----------------
    def reset(self):
        self.state = "menu"
        self.travel = 0.0
        self.dist = 0.0
        self.speed = SPEED_BASE
        self.coins = 0
        self.score = 0
        self.obstacles = []
        self.coin_list = []
        self.particles = []
        self.player.reset()
        self.next_spawn = 55.0
        self.shake = 0.0
        self.dead_t = 0.0
        self.time = 0.0
        self.flash = 0.0
        self.cam_x = 0.0
        self.cam_roll = 0.0
        self.new_record = False

    def start(self):
        self.reset()
        self.state = "playing"
        # 开局先放一组障碍，别让玩家空跑太久
        self.spawn_pattern(76.0)
        self.next_spawn = SPAWN_Z + random.uniform(28.0, 36.0)

    # ---------------- 关卡生成 ----------------
    def add_train(self, lane, z):
        self.obstacles.append(Obstacle(
            "train", LANES[lane], 0.12, z, 2.06, 2.5,
            random.uniform(7.0, 10.5), random.choice(TRAIN_COLORS)))

    def add_barrier(self, lane, z):
        self.obstacles.append(Obstacle(
            "barrier", LANES[lane], 0.0, z, 2.0, 0.86, 0.55, (232, 140, 58)))

    def add_bar(self, lane, z):
        self.obstacles.append(Obstacle(
            "bar", LANES[lane], 1.55, z, 2.0, 1.05, 0.55, (84, 196, 232)))

    def coin_line(self, lane, z, n):
        for i in range(n):
            self.coin_list.append(Coin(LANES[lane], 1.05, z + i * 2.3))

    def coin_arc(self, lane, z, n):
        for i in range(n):
            u = i / max(1, n - 1)
            self.coin_list.append(
                Coin(LANES[lane], 1.0 + 1.15 * math.sin(math.pi * u), z + (i - n * 0.5) * 2.0))

    def spawn_pattern(self, z):
        r = random.random()
        lanes = (0, 1, 2)
        if r < 0.26:
            lane = random.choice(lanes)
            self.add_train(lane, z)
            self.coin_line(random.choice([l for l in lanes if l != lane]), z - 4.0, 5)
        elif r < 0.46:
            free = random.choice(lanes)
            for l in lanes:
                if l != free:
                    self.add_train(l, z + random.uniform(-1.5, 1.5))
            self.coin_line(free, z - 7.0, 9)
        elif r < 0.64:
            blocked = random.sample(lanes, 2)
            for l in blocked:
                self.add_barrier(l, z)
            self.coin_arc(random.choice(blocked), z, 7)
        elif r < 0.80:
            blocked = random.sample(lanes, random.choice((1, 2)))
            for l in blocked:
                self.add_bar(l, z)
            free = [l for l in lanes if l not in blocked]
            self.coin_line(random.choice(free), z - 2.0, 4)
        elif r < 0.92:
            for l in lanes:
                self.add_barrier(l, z)
            self.coin_arc(1, z, 9)
        else:
            self.coin_line(random.choice(lanes), z, 6)

    # ---------------- 世界推进 ----------------
    def scroll(self, dt):
        d = self.speed * dt
        self.travel += d
        self.dist += d

        for ob in self.obstacles:
            ob.z -= d
        for c in self.coin_list:
            c.z -= d

        if self.obstacles:
            self.obstacles = [o for o in self.obstacles if o.z + o.sz * 0.5 > DESPAWN_Z]
        if self.coin_list:
            self.coin_list = [c for c in self.coin_list if c.z > DESPAWN_Z - 4.0]

        self.next_spawn -= d
        if self.next_spawn <= SPAWN_Z:
            self.spawn_pattern(SPAWN_Z)
            self.next_spawn = SPAWN_Z + random.uniform(25.0, 35.0)

        self.speed = min(SPEED_MAX, SPEED_BASE + self.dist * 0.0105)

    def update(self, dt):
        self.time += dt
        if self.flash > 0.0:
            self.flash = max(0.0, self.flash - dt * 3.0)
        if self.shake > 0.0:
            self.shake = max(0.0, self.shake - dt * 2.6)

        if self.state == "menu":
            self.speed = 11.0
            self.travel += self.speed * dt
            self.player.phase += dt * 9.0
            self._update_camera(dt)
            return

        if self.state in ("paused", "dead"):
            if self.state == "dead":
                self.dead_t += dt
                self._update_particles(dt)
            self._update_camera(dt)
            return

        # --- playing ---
        self.scroll(dt)
        self.player.update(dt, self.speed)
        if self.player.just_landed:
            for _ in range(7):
                self.particles.append([
                    self.player.x + random.uniform(-0.35, 0.35),
                    random.uniform(0.05, 0.25),
                    random.uniform(-0.35, 0.35),
                    random.uniform(-2.6, 2.6), random.uniform(0.6, 2.6), random.uniform(-1.2, 1.2),
                    random.uniform(0.22, 0.42), (150, 168, 200),
                ])
        self._check_collisions(dt)
        self._update_particles(dt)
        self._update_camera(dt)

    def _update_camera(self, dt):
        p = self.player
        tgt_x = p.x * 0.62
        k = 1.0 - math.exp(-9.0 * dt)
        self.cam_x += (tgt_x - self.cam_x) * k
        tgt_roll = -p.tilt * 0.10
        self.cam_roll += (tgt_roll - self.cam_roll) * k

        sx = sy = 0.0
        if self.shake > 0.0:
            a = self.shake * self.shake * 0.55
            sx = random.uniform(-a, a)
            sy = random.uniform(-a, a)

        cam_y = 3.15 + p.y * 0.30
        self.cam.look_at((self.cam_x + sx, cam_y + sy, -8.2),
                         (self.cam_x * 1.1, 1.25 + p.y * 0.45, 9.0),
                         self.cam_roll)

    # ---------------- 碰撞 ----------------
    def _check_collisions(self, dt):
        p = self.player
        # 玩家自身 z 向有 0.9 的厚度，单帧位移远小于它，只需很小的扫掠余量
        sweep = min(self.speed * dt * 0.5, 0.35)
        px0, px1 = p.x - 0.40, p.x + 0.40
        py0, py1 = p.y, p.y + p.height
        pz0, pz1 = -0.45 - sweep, 0.45 + sweep

        for ob in self.obstacles:
            if abs(ob.z) > ob.sz * 0.5 + 3.0:
                continue
            ox0, ox1 = ob.x - ob.sx * 0.5, ob.x + ob.sx * 0.5
            oy0, oy1 = ob.y0, ob.y0 + ob.sy
            oz0, oz1 = ob.z - ob.sz * 0.5, ob.z + ob.sz * 0.5
            if px1 > ox0 and px0 < ox1 and py1 > oy0 and py0 < oy1 and pz1 > oz0 and pz0 < oz1:
                self._die(ob)
                return

        for c in self.coin_list:
            if -1.2 < c.z < 1.2 and abs(c.x - p.x) < 0.68 and (p.y - 0.4) < c.y < (p.y + p.height + 0.4):
                self._burst(c.x, c.y, c.z)
                c.z = -999.0
                self.coins += 1
                self.score += 10
                self.flash = 0.42
        if self.coin_list:
            self.coin_list = [c for c in self.coin_list if c.z > -900.0]

    def _die(self, ob):
        self.state = "dead"
        self.dead_t = 0.0
        self.shake = 1.0
        self.score = int(self.dist) + self.coins * 10
        self.new_record = self.score > self.best
        if self.new_record:
            self.best = self.score
            self._save_best()
        p = self.player
        for _ in range(26):
            self.particles.append([
                p.x + random.uniform(-0.4, 0.4),
                p.y + random.uniform(0.2, 1.6),
                random.uniform(-0.4, 0.4),
                random.uniform(-3, 3), random.uniform(2, 9), random.uniform(-3, 3),
                random.uniform(0.5, 1.1), random.choice((200, 240, 255)),
            ])

    def _burst(self, x, y, z):
        for _ in range(9):
            self.particles.append([
                x, y, z,
                random.uniform(-2.4, 2.4), random.uniform(1.0, 5.0), random.uniform(-2.4, 2.4),
                random.uniform(0.25, 0.5), 1,
            ])

    def _update_particles(self, dt):
        alive = []
        for pt in self.particles:
            pt[6] -= dt
            if pt[6] <= 0:
                continue
            pt[3] *= 0.96
            pt[5] *= 0.96
            pt[4] -= 14.0 * dt
            pt[0] += pt[3] * dt
            pt[1] += pt[4] * dt
            pt[2] += pt[5] * dt - (self.speed * dt if self.state == "playing" else 0.0)
            alive.append(pt)
        self.particles = alive

    # ---------------- 绘制 ----------------
    def _make_background(self):
        surf = pygame.Surface((WIDTH, HEIGHT))
        horizon = HEIGHT * 0.41
        for y in range(HEIGHT):
            t = (y / HEIGHT) ** 1.5
            c = mix(SKY_TOP, SKY_BOT, t)
            g = max(0.0, 1.0 - abs(y - horizon) / 150.0) ** 2
            if g > 0.0:
                c = mix(c, HORIZON_GLOW, g * 0.42)
            pygame.draw.line(surf, c, (0, y), (WIDTH, y))
        # 星点
        rnd = random.Random(2024)
        for _ in range(90):
            x = rnd.randrange(WIDTH)
            y = int(rnd.uniform(0, horizon - 20))
            b = rnd.randint(40, 120)
            surf.set_at((x, y), (b, b + 12, b + 30))
        return surf

    def draw(self):
        screen = self.screen
        screen.blit(self.bg, (0, 0))

        r = self.renderer
        r.begin()
        self._draw_ground(r)
        self._draw_walls(r)
        self._draw_coins(r)
        self._draw_obstacles(r)
        self._draw_shadow(r)
        self._draw_player(r)
        r.flush()

        self._draw_particles()
        if self.flash > 0.0:
            self._draw_flash()
        self._draw_hud()

    # --- 地面 ---
    def _draw_ground(self, r):
        seg = 2.0
        off = self.travel % seg
        idx0 = int(self.travel / seg)
        z = -14.0 - off
        for i in range(72):
            zz = z + i * seg
            if zz > 124.0:
                break
            c = GROUND_A if (i + idx0) % 2 == 0 else GROUND_B
            r.quad(((5.15, 0.0, zz), (5.15, 0.0, zz + seg),
                    (-5.15, 0.0, zz + seg), (-5.15, 0.0, zz)), c,
                   (0.0, 1.0, 0.0), bias=2.0)

        # 车道虚线
        seg = 5.2
        off = self.travel % seg
        z = -14.0 - off
        for side in (-1, 1):
            x = side * 1.175
            for i in range(26):
                zz = z + i * seg
                if zz > 118.0:
                    break
                r.quad(((x - 0.07, 0.02, zz), (x + 0.07, 0.02, zz),
                        (x + 0.07, 0.02, zz + 2.6), (x - 0.07, 0.02, zz + 2.6)),
                       C_LINE, (0.0, 1.0, 0.0), bias=1.3, lit=False)

        # 跑道外缘亮带（分成小段，保证画家算法排序正确）
        seg = 6.0
        off = self.travel % seg
        z = -14.0 - off
        for side in (-1, 1):
            x = side * (TRACK_HALF - 0.16)
            for i in range(22):
                zz = z + i * seg
                if zz > 118.0:
                    break
                r.quad(((x - 0.10, 0.03, zz), (x + 0.10, 0.03, zz),
                        (x + 0.10, 0.03, zz + seg - 0.4),
                        (x - 0.10, 0.03, zz + seg - 0.4)),
                       (70, 90, 126), (0.0, 1.0, 0.0), bias=1.25, lit=False)

    # --- 两侧墙体 ---
    def _draw_walls(self, r):
        seg = 6.0
        off = self.travel % seg
        idx0 = int(self.travel / seg)
        zbase = -14.0 - off
        for side in (-1, 1):
            x = side * (TRACK_HALF + 0.35)
            nx = -side
            for i in range(24):
                z = zbase + i * seg
                if z > 120.0:
                    break
                alt = (i + idx0) % 2 == 0
                c = WALL_A if alt else WALL_B
                r.quad(((x, 0.45, z), (x, 0.45, z + seg),
                        (x, 3.6, z + seg), (x, 3.6, z)),
                       c, (nx, 0.0, 0.0), bias=0.6)
                # 踢脚
                r.quad(((x, 0.0, z), (x, 0.0, z + seg),
                        (x, 0.45, z + seg), (x, 0.45, z)),
                       (54, 60, 80), (nx, 0.0, 0.0), bias=0.62)
                # 分缝
                r.quad(((x - side * 0.01, 0.45, z), (x - side * 0.01, 0.45, z + 0.16),
                        (x - side * 0.01, 3.6, z + 0.16), (x - side * 0.01, 3.6, z)),
                       (50, 56, 74), (nx, 0.0, 0.0), bias=0.58)
                # 顶部灯槽
                r.quad(((x - side * 0.03, 3.30, z + 0.35), (x - side * 0.03, 3.30, z + seg - 0.35),
                        (x - side * 0.03, 3.52, z + seg - 0.35), (x - side * 0.03, 3.52, z + 0.35)),
                       (118, 110, 88) if alt else (96, 90, 74),
                       (nx, 0.0, 0.0), bias=0.55, lit=False)
                # 霓虹灯条
                if (i + idx0) % 3 == 0:
                    nc = C_NEON[(i + idx0) // 3 % 2]
                    r.quad(((x - side * 0.03, 2.56, z + 0.9), (x - side * 0.03, 2.56, z + seg - 1.4),
                            (x - side * 0.03, 2.68, z + seg - 1.4), (x - side * 0.03, 2.68, z + 0.9)),
                           nc, (nx, 0.0, 0.0), bias=0.55, lit=False)

    # --- 障碍物 ---
    def _draw_obstacles(self, r):
        for ob in self.obstacles:
            if ob.z + ob.sz * 0.5 < -12.0 or ob.z - ob.sz * 0.5 > 124.0:
                continue

            # 接地阴影
            self._soft_shadow(r, ob.x, ob.z, ob.sx * 0.5 + 0.16, ob.sz * 0.5 + 0.16)

            if ob.kind == "train":
                cy = ob.y0 + ob.sy * 0.5
                # 底盘
                r.box(ob.x, 0.16, ob.z, ob.sx + 0.03, 0.32, ob.sz, (26, 30, 44))
                # 车身
                r.box(ob.x, cy, ob.z, ob.sx, ob.sy, ob.sz, ob.color)
                # 车窗带（更宽，避免共面闪烁）
                r.box(ob.x, 1.92, ob.z, ob.sx + 0.07, 0.62, ob.sz - 0.55, (34, 48, 74))
                # 车顶
                r.box(ob.x, ob.y0 + ob.sy + 0.05, ob.z, ob.sx + 0.05, 0.16,
                      ob.sz - 0.12, (228, 232, 240))
                # 侧腰线
                r.box(ob.x, 1.30, ob.z, ob.sx + 0.05, 0.10, ob.sz - 0.2, (245, 245, 250))
                # 车头灯
                zf = ob.z - ob.sz * 0.5 - 0.03
                r.box(ob.x, 0.78, zf, ob.sx - 0.16, 0.46, 0.06, (26, 30, 44))
                for s in (-1, 1):
                    r.box(ob.x + s * 0.55, 0.78, zf - 0.03, 0.30, 0.20, 0.07, (255, 238, 176))
            elif ob.kind == "barrier":
                r.box(ob.x, ob.y0 + ob.sy * 0.5, ob.z, ob.sx, ob.sy, ob.sz, ob.color)
                r.box(ob.x, 0.30, ob.z, ob.sx + 0.05, 0.20, ob.sz + 0.05, (250, 250, 250))
                r.box(ob.x, 0.62, ob.z, ob.sx + 0.05, 0.18, ob.sz + 0.05, (208, 62, 62))
                r.box(ob.x, ob.y0 + ob.sy - 0.04, ob.z, ob.sx + 0.05, 0.12, ob.sz + 0.05,
                      (252, 208, 122))
            else:  # bar（需要翻滚通过）
                for s in (-1, 1):
                    r.box(ob.x + s * (ob.sx * 0.5 + 0.11), 1.30, ob.z, 0.22, 2.6, 0.26,
                          (126, 140, 172))
                r.box(ob.x, ob.y0 + ob.sy * 0.5, ob.z, ob.sx, ob.sy, ob.sz, ob.color)
                r.box(ob.x, ob.y0 + 0.10, ob.z, ob.sx + 0.05, 0.20, ob.sz + 0.06, (236, 250, 255))
                r.box(ob.x, ob.y0 + ob.sy - 0.05, ob.z, ob.sx + 0.05, 0.14, ob.sz + 0.06,
                      (46, 92, 118))

    # --- 金币 ---
    def _draw_coins(self, r):
        for c in self.coin_list:
            if c.z < -3.0 or c.z > 116.0:
                continue
            sp = abs(math.cos(c.spin + self.time * 4.2))
            rx = 0.34 * (0.14 + 0.86 * sp)
            r.disc(c.x, c.y, c.z, rx, 0.34, (168, 118, 18), 10, bias=0.02)
            r.disc(c.x, c.y, c.z, rx * 0.92, 0.31, C_GOLD, 10, bias=0.0)
            if sp > 0.30:
                r.disc(c.x, c.y, c.z, rx * 0.48, 0.155, C_GOLD2, 8, bias=-0.03)

    # --- 柔和地面阴影（三层叠出渐变感）---
    def _soft_shadow(self, r, x, z, hx, hz):
        for k, col, bias in ((1.00, (38, 43, 57), 1.90),
                             (0.70, (31, 36, 49), 1.88),
                             (0.42, (25, 29, 40), 1.86)):
            a = hx * k
            b = hz * k
            r.quad(((x - a, 0.015, z - b), (x + a, 0.015, z - b),
                    (x + a, 0.015, z + b), (x - a, 0.015, z + b)),
                   col, (0.0, 1.0, 0.0), bias=bias, lit=False)

    # --- 玩家阴影 ---
    def _draw_shadow(self, r):
        p = self.player
        s = 0.62 * max(0.40, 1.0 - p.y * 0.26)
        self._soft_shadow(r, p.x, 0.1, s, s * 0.9)

    # --- 角色 ---
    def _draw_player(self, r):
        p = self.player
        x, y = p.x, p.y
        lean = clamp(p.tilt * 0.35, -0.18, 0.18)

        if p.rolling > 0.0:
            # 翻滚：整体压低
            parts = (
                (-0.16, 0.32, 0.40, 0.21, 0.50, 0.26, C_JEANS),
                (0.16, 0.32, 0.40, 0.21, 0.50, 0.26, C_JEANS),
                (-0.16, 0.12, 0.50, 0.24, 0.14, 0.30, C_SHOE),
                (0.16, 0.12, 0.50, 0.24, 0.14, 0.30, C_SHOE),
                (0.00, 0.44, 0.02, 0.66, 0.56, 0.62, C_HOODIE),
                (-0.34, 0.42, -0.02, 0.16, 0.40, 0.18, C_HOODIE),
                (0.34, 0.42, -0.02, 0.16, 0.40, 0.18, C_HOODIE),
                (0.00, 0.42, -0.36, 0.48, 0.38, 0.20, C_PACK),
                (0.00, 0.74, 0.30, 0.42, 0.36, 0.36, C_SKIN),
                (0.00, 0.88, 0.28, 0.46, 0.12, 0.42, C_CAP),
            )
        else:
            swing = math.sin(p.phase)
            if y > 0.05:                      # 空中姿态
                ls, rs, la, ra = 0.28, -0.20, -0.24, 0.22
            else:
                ls, rs = swing * 0.58, -swing * 0.58
                la, ra = -swing * 0.42, swing * 0.42
            parts = (
                (-0.155, 0.40, ls * 0.45, 0.21, 0.80, 0.25, C_JEANS),
                (0.155, 0.40, rs * 0.45, 0.21, 0.80, 0.25, C_JEANS),
                (-0.155, 0.09, ls * 0.58 + 0.05, 0.24, 0.15, 0.33, C_SHOE),
                (0.155, 0.09, rs * 0.58 + 0.05, 0.24, 0.15, 0.33, C_SHOE),
                (0.00, 0.80, 0.00, 0.56, 0.18, 0.38, C_JEANS),     # 胯
                (0.00, 1.12, 0.00, 0.60, 0.62, 0.40, C_HOODIE),    # 躯干
                (-0.40, 1.34, la * 0.5, 0.17, 0.48, 0.17, C_HOODIE),
                (0.40, 1.34, ra * 0.5, 0.17, 0.48, 0.17, C_HOODIE),
                (0.00, 1.42, -0.08, 0.50, 0.15, 0.36, C_HOODIE),   # 帽兜
                (0.00, 1.24, -0.27, 0.42, 0.40, 0.18, C_PACK),     # 背包
                (0.00, 1.64, 0.00, 0.44, 0.42, 0.42, C_SKIN),      # 头
                (0.00, 1.87, 0.02, 0.48, 0.12, 0.46, C_CAP),       # 帽
                (0.00, 1.83, 0.24, 0.42, 0.07, 0.20, C_CAP),       # 帽檐
            )

        for ox, cy, cz, sx, sy, sz, col in parts:
            r.box(x + ox + lean, y + cy, cz, sx, sy, sz, col)

    # --- 粒子 / 闪光 ---
    def _draw_particles(self):
        if not self.particles:
            return
        cam = self.cam
        surf = self.screen
        for pt in self.particles:
            c = cam.to_cam((pt[0], pt[1], pt[2]))
            if c[2] < NEAR:
                continue
            sx = WIDTH * 0.5 + FOCAL * c[0] / c[2]
            sy = HEIGHT * 0.5 - FOCAL * c[1] / c[2]
            rad = max(1, int(FOCAL * 0.07 * pt[6] / c[2] * 6))
            if rad > 12:
                rad = 12
            col = pt[7]
            if isinstance(col, int):
                col = (255, 220, 120)
            pygame.draw.circle(surf, col, (int(sx), int(sy)), rad)

    def _draw_flash(self):
        a = int(52 * self.flash)
        if a <= 0:
            return
        self.flash_surf.set_alpha(a)
        self.screen.blit(self.flash_surf, (0, 0))

    # --- HUD ---
    def _text(self, txt, font, color, pos, anchor="topleft", shadow=True):
        img = font.render(txt, True, color)
        rect = img.get_rect()
        setattr(rect, anchor, pos)
        if shadow:
            sh = font.render(txt, True, (0, 0, 0))
            self.screen.blit(sh, (rect.x + 2, rect.y + 2))
        self.screen.blit(img, rect)
        return rect

    def _draw_hud(self):
        if self.state == "menu":
            self._draw_menu()
            return
        if self.state == "dead":
            self._draw_gameover()
            return

        score = int(self.dist) + self.coins * 10

        self._text(T("分数", "SCORE"), self.f_label, (128, 148, 184), (36, 24))
        self._text(f"{score:06d}", self.f_score, (245, 250, 255), (32, 44))
        self._text(T("最高", "BEST") + f"  {self.best}", self.f_label,
                   (128, 148, 184), (36, 104))

        # 金币
        cx = WIDTH - 132
        pygame.draw.circle(self.screen, (168, 118, 18), (cx, 53), 16)
        pygame.draw.circle(self.screen, C_GOLD, (cx, 52), 15)
        pygame.draw.circle(self.screen, C_GOLD2, (cx, 52), 7)
        self._text(str(self.coins), self.f_mid, (255, 240, 190), (cx + 28, 34))

        # 速度条
        bx, by = 36, HEIGHT - 46
        pygame.draw.rect(self.screen, (36, 44, 62), (bx, by, 180, 8), border_radius=4)
        w = int(180 * clamp((self.speed - SPEED_BASE) / (SPEED_MAX - SPEED_BASE), 0, 1))
        if w > 2:
            pygame.draw.rect(self.screen, (92, 200, 240), (bx, by, w, 8), border_radius=4)
        self._text(f"{self.speed * 3.6:.0f} km/h", self.f_label, (120, 140, 176),
                   (bx + 190, by - 8))

        if self.state == "paused":
            self._text(T("暂停中", "PAUSED"), self.f_big, (230, 240, 255),
                       (WIDTH // 2, HEIGHT // 2), "center")
            self._text(T("按 P 继续", "press P to resume"), self.f_small,
                       (150, 168, 200), (WIDTH // 2, HEIGHT // 2 + 58), "center")

    def _draw_menu(self):
        self.screen.blit(self.overlay, (0, 0))
        cy = HEIGHT // 2 - 40
        self._text(T("地 铁 跑 酷", "SUBWAY  RUN"), self.f_big, (255, 255, 255),
                   (WIDTH // 2, cy - 150), "center")
        self._text(T("3D · Python 纯手写渲染 · 无尽跑酷", "3D endless runner · pure Python"),
                   self.f_small, (126, 194, 232), (WIDTH // 2, cy - 78), "center")

        pulse = 0.6 + 0.4 * math.sin(self.time * 3.4)
        col = mix((90, 110, 150), (255, 236, 170), pulse)
        self._text(T("按 空格 开始", "PRESS  SPACE  TO  START"), self.f_mid, col,
                   (WIDTH // 2, cy + 150), "center")

        hint = T("← → 换道     ↑ 跳跃     ↓ 翻滚     P 暂停",
                 "A/D or ARROWS  switch lanes     SPACE  jump     S  roll")
        self._text(hint, self.f_small, (146, 164, 198), (WIDTH // 2, cy + 208), "center")
        self._text(T("最高分 " + str(self.best), "BEST  " + str(self.best)), self.f_label,
                   (110, 128, 164), (WIDTH // 2, cy + 244), "center")

    def _draw_gameover(self):
        a = int(min(1.0, self.dead_t * 3.2) * 200)
        ov = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        ov.fill((8, 10, 18, a))
        self.screen.blit(ov, (0, 0))
        cy = HEIGHT // 2
        self._text(T("撞 上 了 !", "CRASHED!"), self.f_big, (255, 118, 108),
                   (WIDTH // 2, cy - 120), "center")
        self._text(T("本次得分", "SCORE") + f"   {self.score}", self.f_mid,
                   (240, 246, 255), (WIDTH // 2, cy - 30), "center")
        if self.new_record:
            pulse = 0.5 + 0.5 * math.sin(self.time * 6.0)
            self._text(T("★ 新 纪 录 ★", "* NEW  RECORD *"), self.f_small,
                       mix((255, 200, 80), (255, 255, 220), pulse),
                       (WIDTH // 2, cy + 58), "center")
        self._text(T("金币", "COINS") + f"   {self.coins}        " +
                   T("最高", "BEST") + f"   {self.best}",
                   self.f_small, (150, 170, 205), (WIDTH // 2, cy + 22), "center")
        if self.dead_t > 0.35:
            pulse = 0.5 + 0.5 * math.sin(self.time * 4.5)
            col = mix((100, 120, 160), (255, 236, 170), pulse)
            self._text(T("按 R 重新开始", "PRESS  R  TO  RETRY"), self.f_mid, col,
                       (WIDTH // 2, cy + 100), "center")

    # ---------------- 输入 ----------------
    def handle_event(self, e):
        if e.type == pygame.QUIT:
            return False
        if e.type != pygame.KEYDOWN:
            return True
        k = e.key

        if k == pygame.K_ESCAPE:
            return False

        if self.state == "menu":
            if k in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_UP, pygame.K_w):
                self.start()
        elif self.state == "playing":
            if k in (pygame.K_LEFT, pygame.K_a):
                self.player.move(-1)
            elif k in (pygame.K_RIGHT, pygame.K_d):
                self.player.move(1)
            elif k in (pygame.K_UP, pygame.K_w, pygame.K_SPACE):
                self.player.jump()
            elif k in (pygame.K_DOWN, pygame.K_s):
                self.player.roll()
            elif k == pygame.K_p:
                self.state = "paused"
        elif self.state == "paused":
            if k == pygame.K_p:
                self.state = "playing"
            elif k == pygame.K_r:
                self.start()
        elif self.state == "dead":
            if self.dead_t > 0.35 and k in (pygame.K_r, pygame.K_SPACE, pygame.K_RETURN):
                self.start()
        return True

    # ---------------- 主循环 ----------------
    def run(self):
        while self.running:
            dt = min(self.clock.tick(FPS) / 1000.0, 1.0 / 30.0)
            for e in pygame.event.get():
                if not self.handle_event(e):
                    self.running = False
                    break
            if not self.running:
                break
            self.update(dt)
            self.draw()
            pygame.display.flip()
        pygame.quit()


def main():
    Game().run()


if __name__ == "__main__":
    main()
