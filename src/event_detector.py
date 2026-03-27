import numpy as np
from collections import defaultdict
from .models import ObjectStats, Event, Event, PitchPoint, Track
from .homography import Homography

# all distances in meters, speeds in m/s, accelerations in m/s^2
BALL_POSSESSION_RADIUS = 1.2 # if ball is within this distance, we consider it "possessed"
DUEL_DISTANCE = 1.5 # if two players are within this distance, we consider it a "duel"
TACKLE_DECEL_THRESHOLD = 3.0 # if a player decelerates by this much after a duel, we consider it a "tackle"
DRIBBLE_DIRECTION_DEG  = 45 # if a player changes direction by this much while in possession, we consider it a "dribble"
DRIBBLE_OPPONENT_RADIUS = 2.0 # if an opponent is within this distance during a dribble, we consider it a "dribble with pressure"
SAVE_GOAL_MARGIN_M = 2.0
PASS_MAX_FRAMES = 30 # if the ball moves from one player to another within this many frames, we consider it a "pass"
DUEL_OPPONENT_RADIUS = 3.0 # if an opponent is within this distance during a duel, we consider it a "duel with pressure"

class EventDetector:
    def __init__(self, config: dict, stats_manager):
        self.stats = stats_manager
        self.fps = config["fps"]
        self.events: list[Event] = []


        # rolling stat - keyed by track_id 
        self._in_duel: dict[int, int] = {}
        self._ball_owner: int | None = None 
        self._ball_owner_frame: int = 0
        self._prev_ball_pos: PitchPoint | None = None
        self._prev_positions: dict[int, tuple] = {} # track_id -> (x, y)

    def update(self,
               tracks: list[Track],
               ball: object | None,
               pitch_positions: dict[int, PitchPoint],
               meta):
        
        ball_pos = self._get_ball_pitch_pos(ball)
        self._detect_possession(tracks, ball_pos, meta)
        self._detect_duels_and_tackles(tracks, pitch_positions, meta)
        self._detect_dribbles(tracks, pitch_positions, meta)
        self._detect_passes(tracks, ball_pos, pitch_positions, meta)
        self._detect_saves(tracks, ball_pos, meta)

        self._prev_ball_pos = ball_pos
        self._prev_positions = {
            track.track_id: (pitch_positions[track.track_id].x_m, pitch_positions[track.track_id].y_m)
            for track in tracks if track.track_id in pitch_positions
        }

    def _detect_possession(self, tracks, ball_pos, meta):
        if ball_pos is None:
            return
        
        closest_id = None
        closest_dist = float('inf')

        for track in tracks:
            stats = self.stats.get(track.track_id)
            if not stats or not stats.positions:
                continue
            _, px, py = stats.positions[-1]
            dist = np.hypot(px - ball_pos.x_m, py - ball_pos.y_m)
            if dist < closest_dist:
                closest_dist = dist
                closest_id = track.track_id

        if closest_dist <= BALL_POSSESSION_RADIUS:
            self._ball_owner = closest_id
            self._ball_owner_frame = meta.frame_number
        else:
            self._ball_owner = None

    def _detect_duels_and_tackles(self, tracks, pitch_positions, meta):
        players = [t for t in tracks if t.class_name == "player"]
        for i, t1 in enumerate(players):
            for t2 in players[1+i:]:

                # must be opposite teams 
                s1 = self.stats.get(t1.track_id)
                s2 = self.stats.get(t2.track_id)
                if not s1 or not s2:
                    continue
                if s1.team == s2.team or s1.team == -1 or s2.team == -1:
                    continue

                p1 = pitch_positions.get(t1.track_id)
                p2 = pitch_positions.get(t2.track_id)
                if not p1 or not p2:
                    continue

                dist = np.hypot(p1.x_m - p2.x_m, p1.y_m - p2.y_m)

                if dist <= DUEL_DISTANCE:
                    # one of them must be near the ball
                    if self._ball_owner in (t1.track_id, t2.track_id):
                        continue

                    # check if tackle - sharp decel on ball carrier 
                    ball_carrier = self._ball_owner
                    carrier_stats = self.stats.get(ball_carrier)

                    if (carrier_stats and 
                        len(carrier_stats.accelerations) > 0 and
                        carrier_stats.accelerations[-1] < -TACKLE_DECEL_THRESHOLD):
                        
                        self._fire(Event(
                            event_type="tackle",
                            frame=meta.frame_number,
                            timestamp_s=meta.timestamp_s,
                            primary_track_id=t2.track_id if t2.track_id != ball_carrier else t1.track_id,
                            secondary_track_id=ball_carrier,
                            ball_pos = (self._prev_ball_pos.x_m, self._prev_ball_pos.y_m) if self._prev_ball_pos else None,
                            confidence=min(1.0, DUEL_DISTANCE / max(dist, 1))
                        ))
                    else:
                        duel_key = tuple(sorted((t1.track_id, t2.track_id)))
                        self.in_duel.pop(duel_key, None) # end any existing duel between these players

    def _detect_dribbles(self, tracks, pitch_positions, meta):
        if self._ball_owner is None:
            return

        owner_stats = self.stats.get(self._ball_owner)
        if not owner_stats or len(owner_stats.positions) < 3:
            return

        # direction vectors over last 3 positions
        _, x0, y0 = owner_stats.positions[-3]
        _, x1, y1 = owner_stats.positions[-2]
        _, x2, y2 = owner_stats.positions[-1]

        v1 = np.array([x1 - x0, y1 - y0])
        v2 = np.array([x2 - x1, y2 - y1])

        # need some minimum movement to avoid noise
        if np.linalg.norm(v1) < 0.3 or np.linalg.norm(v2) < 0.3:
            return

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))

        if angle_deg < DRIBBLE_DIRECTION_DEG:
            return   # not enough direction change

        # opponent must be nearby
        owner_pos = pitch_positions.get(self._ball_owner)
        if not owner_pos:
            return

        owner_team = owner_stats.team
        opponent_nearby = any(
            self.stats.get(t.track_id) and
            self.stats.get(t.track_id).team != owner_team and
            self.stats.get(t.track_id).team != -1 and
            np.hypot(
                pitch_positions[t.track_id].x_m - owner_pos.x_m,
                pitch_positions[t.track_id].y_m - owner_pos.y_m
            ) <= DUEL_OPPONENT_RADIUS
            for t in tracks
            if t.class_name == "player" and t.track_id in pitch_positions
        )

        if opponent_nearby:
            self._fire(Event(
                event_type       = "dribble",
                frame            = meta.frame_number,
                timestamp_s      = meta.timestamp_s,
                primary_track_id = self._ball_owner,
                secondary_track_id = None,
                ball_pos         = (owner_pos.x_m, owner_pos.y_m),
                confidence       = min(1.0, angle_deg / 90),
            ))

    def _detect_saves(self, tracks, ball_pos, meta):
        if ball_pos is None or self._prev_ball_pos is None:
            return

        # check both goal mouths (x≈0 and x≈105)
        for goal_x in [0.0, 105.0]:
            goal_centre = np.array([goal_x, 34.0])
            ball_now    = np.array([ball_pos.x_m, ball_pos.y_m])
            ball_prev   = np.array([self._prev_ball_pos.x_m,
                                    self._prev_ball_pos.y_m])

            # ball was moving toward goal and is now close to goal line
            moving_toward = np.dot(ball_now - ball_prev,
                                   goal_centre - ball_prev) > 0
            near_goal     = abs(ball_pos.x_m - goal_x) < SAVE_GOAL_MARGIN_M

            if not (moving_toward and near_goal):
                continue

            # find the nearest player to the ball in this area
            for track in tracks:
                if track.class_name != "player":
                    continue
                stats = self.stats.get(track.track_id)
                if not stats:
                    continue
                if not stats.positions:
                    continue
                _, px, py = stats.positions[-1]
                if np.hypot(px - ball_pos.x_m, py - ball_pos.y_m) < 2.0:
                    self._fire(Event(
                        event_type        = "save",
                        frame             = meta.frame_number,
                        timestamp_s       = meta.timestamp_s,
                        primary_track_id  = track.track_id,
                        secondary_track_id= None,
                        ball_pos          = (ball_pos.x_m, ball_pos.y_m),
                        confidence        = 0.75,
                    ))
                    break

    def _detect_passes(self, tracks, ball_pos, pitch_positions, meta):
        if ball_pos is None or self._ball_owner is None:
            return

        prev_owner = self._ball_owner
        frames_held = meta.frame_number - self._ball_owner_frame

        # find new owner this frame
        new_owner = None
        for track in tracks:
            if track.track_id == prev_owner:
                continue
            pos = pitch_positions.get(track.track_id)
            if not pos:
                continue
            if np.hypot(pos.x_m - ball_pos.x_m,
                        pos.y_m - ball_pos.y_m) <= BALL_POSSESSION_RADIUS:
                new_owner = track.track_id
                break

        if new_owner is None:
            return

        s1 = self.stats.get(prev_owner)
        s2 = self.stats.get(new_owner)
        if not s1 or not s2:
            return

        # must be same team and transfer must happen within PASS_MAX_FRAMES
        if s1.team == s2.team and frames_held <= PASS_MAX_FRAMES:
            self._fire(Event(
                event_type        = "pass",
                frame             = meta.frame_number,
                timestamp_s       = meta.timestamp_s,
                primary_track_id  = prev_owner,
                secondary_track_id= new_owner,
                ball_pos          = (ball_pos.x_m, ball_pos.y_m),
                confidence        = 0.8,
            ))
    
    def _fire(self, event: Event):
        self.events.append(event)
        # attach to the player's own stats record so output_writer sees it
        stats = self.stats.get(event.primary_track_id)
        if stats:
            stats.events.append({
                "type":       event.event_type,
                "frame":      event.frame,
                "timestamp_s": event.timestamp_s,
                "confidence": event.confidence,
            })

    def _get_ball_pitch_pos(self, ball) -> PitchPoint | None:
        # ball is a Detection | None from detector.py
        # its pitch position is passed in already resolved upstream
        return self._prev_ball_pos   # placeholder — see pipeline note below