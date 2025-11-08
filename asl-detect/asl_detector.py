import math


class ASLDetector:
    def __init__(self):
        # Finger tip and pip (proximal interphalangeal) landmark indices
        self.tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        self.pip_ids = [3, 6, 10, 14, 18]  # Corresponding PIP joints
        self.dip_ids = [3, 7, 11, 15, 19]  # DIP joints
        self.mcp_ids = [2, 5, 9, 13, 17]  # MCP joints (knuckles)

    def get_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def get_angle(self, p1, p2, p3):
        """Calculate angle at p2 formed by p1-p2-p3"""
        a = self.get_distance(p2, p3)
        b = self.get_distance(p1, p2)
        c = self.get_distance(p1, p3)

        if a * b == 0:
            return 0

        try:
            angle = math.acos((a**2 + b**2 - c**2) / (2 * a * b))
            return math.degrees(angle)
        except:
            return 0

    def palm_size(self, landmarks):
        """Calculate palm size for normalization"""
        wrist = landmarks[0]
        mid_mcp = landmarks[9]
        idx_mcp = landmarks[5]
        pky_mcp = landmarks[17]
        # Use diagonal + width for stability
        base = self.get_distance((wrist.x, wrist.y), (mid_mcp.x, mid_mcp.y))
        width = self.get_distance((idx_mcp.x, idx_mcp.y), (pky_mcp.x, pky_mcp.y))
        return max(1e-6, (base + width) / 2.0)

    def angle_between(self, a, b, c):
        """Angle at b for vectors ba and bc"""
        return self.get_angle((a.x, a.y), (b.x, b.y), (c.x, c.y))

    def is_thumb_extended_clear(self, landmarks):
        """Check if thumb is clearly extended (>135° at MCP)"""
        tip, mcp, cmc = landmarks[4], landmarks[2], landmarks[1]
        ang = self.get_angle((cmc.x, cmc.y), (mcp.x, mcp.y), (tip.x, tip.y))
        return ang > 135

    def infer_right_hand(self, landmarks):
        """Infer if this is a right hand (image already flipped)"""
        return landmarks[5].x < landmarks[17].x

    def is_finger_extended(self, landmarks, finger_idx):
        """Check if a finger is fully extended using angles"""
        if finger_idx == 0:  # Thumb - different check
            tip = landmarks[4]
            ip = landmarks[3]
            mcp = landmarks[2]
            cmc = landmarks[1]

            # Check if thumb is extended outward
            angle = self.get_angle((cmc.x, cmc.y), (mcp.x, mcp.y), (tip.x, tip.y))
            return angle > 130  # Relatively straight
        else:
            # For other fingers, check if they're extended (tips above mcp)
            tip = landmarks[self.tip_ids[finger_idx]]
            pip = landmarks[self.pip_ids[finger_idx]]
            mcp = landmarks[self.mcp_ids[finger_idx]]

            # Check angle at PIP joint
            angle = self.get_angle((mcp.x, mcp.y), (pip.x, pip.y), (tip.x, tip.y))

            # Extended if angle > 140 degrees (relatively straight)
            # Also check vertical position
            return angle > 140 and tip.y < mcp.y

    def is_finger_curled(self, landmarks, finger_idx):
        """Check if finger is curled/bent"""
        if finger_idx == 0:  # Thumb
            tip = landmarks[4]
            mcp = landmarks[2]
            wrist = landmarks[0]

            # Thumb curled if tip is close to palm
            tip_to_wrist = self.get_distance((tip.x, tip.y), (wrist.x, wrist.y))
            mcp_to_wrist = self.get_distance((mcp.x, mcp.y), (wrist.x, wrist.y))
            return tip_to_wrist < mcp_to_wrist * 1.3
        else:
            tip = landmarks[self.tip_ids[finger_idx]]
            pip = landmarks[self.pip_ids[finger_idx]]
            mcp = landmarks[self.mcp_ids[finger_idx]]

            # Check if tip is below or at same level as mcp (curled)
            angle = self.get_angle((mcp.x, mcp.y), (pip.x, pip.y), (tip.x, tip.y))
            return angle < 130 or tip.y >= mcp.y

    def detect_asl_letter(self, landmarks):
        """Detect ASL letter based on hand landmarks"""
        if not landmarks:
            return None

        # Calculate palm-scaled thresholds
        ps = self.palm_size(landmarks)
        near = 0.10 * ps
        tight = 0.14 * ps
        loose = 0.22 * ps
        wide = 0.28 * ps

        # Get extension status for each finger
        extended = [self.is_finger_extended(landmarks, i) for i in range(5)]
        curled = [self.is_finger_curled(landmarks, i) for i in range(5)]

        # Get key landmark positions
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]

        # Calculate key distances
        thumb_index_tip = self.get_distance(
            (thumb_tip.x, thumb_tip.y), (index_tip.x, index_tip.y)
        )
        thumb_middle_tip = self.get_distance(
            (thumb_tip.x, thumb_tip.y), (middle_tip.x, middle_tip.y)
        )
        thumb_index_mcp = self.get_distance(
            (thumb_tip.x, thumb_tip.y), (index_mcp.x, index_mcp.y)
        )
        index_middle_tip = self.get_distance(
            (index_tip.x, index_tip.y), (middle_tip.x, middle_tip.y)
        )

        # === LETTER DETECTION (in priority order) ===

        # O - all fingertips near thumb (circle) - CHECK BEFORE A
        if all(
            self.get_distance(
                (landmarks[4].x, landmarks[4].y), (landmarks[tip].x, landmarks[tip].y)
            )
            < near
            for tip in [8, 12, 16, 20]
        ):
            return "O"

        # A - Closed fist with thumb extended to side
        if extended[0] and all(curled[1:]):
            return "A"

        # B - All four fingers extended straight up, thumb tucked across palm
        if all(extended[1:]) and curled[0]:
            # Check fingers are close together (not spread) - palm-scaled
            idx_mid_close = (
                self.get_distance(
                    (index_tip.x, index_tip.y), (middle_tip.x, middle_tip.y)
                )
                < tight
            )
            mid_ring_close = (
                self.get_distance(
                    (middle_tip.x, middle_tip.y), (ring_tip.x, ring_tip.y)
                )
                < tight
            )

            # Check thumb is tucked (close to index mcp/palm) - palm-scaled
            thumb_tucked = thumb_index_mcp < tight

            if idx_mid_close and mid_ring_close and thumb_tucked:
                return "B"

        # C - curved hand shape with true curvature + opening
        pip_angles = []
        for f in [1, 2, 3, 4]:
            pip = landmarks[self.pip_ids[f]]
            mcp = landmarks[self.mcp_ids[f]]
            tip = landmarks[self.tip_ids[f]]
            pip_angles.append(
                self.get_angle((mcp.x, mcp.y), (pip.x, pip.y), (tip.x, tip.y))
            )

        curved = all(80 <= a <= 150 for a in pip_angles)  # not straight, not tight curl
        opening = (
            self.get_distance(
                (landmarks[8].x, landmarks[8].y), (landmarks[20].x, landmarks[20].y)
            )
            > 0.6 * ps
        )
        not_closed = thumb_index_tip > 0.18 * ps
        if curved and opening and not_closed:
            return "C"

        # F - Index finger curled touching thumb, others extended
        if extended[2] and extended[3] and extended[4]:
            # Index should be partially curled with tip near thumb
            index_curled_at_thumb = thumb_index_tip < near and not extended[1]
            if index_curled_at_thumb:
                return "F"

        # D - Index extended, thumb touching middle finger, others curled
        if extended[1] and curled[2] and curled[3] and curled[4]:
            # Thumb should touch the middle finger
            if thumb_middle_tip < near:
                return "D"

        # I - Only pinky extended; allow thumb neutral but not clearly extended
        thumb_clear = self.is_thumb_extended_clear(landmarks)
        if extended[4] and all(curled[1:4]) and not thumb_clear:
            return "I"

        # Y - Thumb and pinky clearly extended (shaka/hang loose)
        if (
            extended[0]
            and extended[4]
            and all(curled[1:4])
            and self.is_thumb_extended_clear(landmarks)
        ):
            return "Y"

        # L - Thumb and index extended at 90 degrees, others curled
        if extended[0] and extended[1] and all(curled[2:]):
            # Angle at wrist between thumb dir and index dir (90° ± 30°)
            ang = self.angle_between(thumb_tip, wrist, index_tip)
            if 60 <= ang <= 120:
                return "L"

        # V - Index and middle extended and SEPARATED (peace sign)
        if extended[1] and extended[2] and all(curled[3:]):
            # Fingers must be spread apart - palm-scaled
            if index_middle_tip > 0.17 * ps:  # more generous spread
                index_up = index_tip.y < index_mcp.y
                middle_up = middle_tip.y < middle_mcp.y
                if index_up and middle_up:
                    return "V"

        # U - Index and middle extended and TOGETHER
        if extended[1] and extended[2] and all(curled[3:]):
            # Fingers must be close together - palm-scaled
            if index_middle_tip < 0.11 * ps:  # scaled closeness
                index_up = index_tip.y < index_mcp.y
                middle_up = middle_tip.y < middle_mcp.y
                if index_up and middle_up:
                    return "U"

        # W - Index, middle, and ring extended
        if (
            extended[1]
            and extended[2]
            and extended[3]
            and curled[4]
            and (curled[0] or not extended[0])
        ):
            return "W"

        # R - Index and middle extended, middle crosses over index (with handedness)
        if extended[1] and extended[2] and all(curled[3:]):
            right = self.infer_right_hand(landmarks)
            crossed = (
                (middle_tip.x < index_tip.x) if right else (middle_tip.x > index_tip.x)
            )
            very_close = index_middle_tip < 0.08 * ps
            if crossed and very_close:
                return "R"

        # K - Index up, middle angled, thumb contacts middle (not index)
        if extended[1] and extended[2] and all(curled[3:]):
            # Thumb near middle tip; not near index tip
            thumb_middle_close = (
                self.get_distance(
                    (thumb_tip.x, thumb_tip.y), (middle_tip.x, middle_tip.y)
                )
                < 0.14 * ps
            )
            thumb_index_far = (
                self.get_distance(
                    (thumb_tip.x, thumb_tip.y), (index_tip.x, index_tip.y)
                )
                > 0.18 * ps
            )

            # Angle between index and middle not ~0 (they shouldn't be parallel)
            v_sep = abs(index_tip.x - middle_tip.x) + abs(index_tip.y - middle_tip.y)

            # Check angle at wrist between index and middle
            ang_im = self.angle_between(index_tip, wrist, middle_tip)

            if (
                thumb_middle_close
                and thumb_index_far
                and v_sep > 0.12 * ps
                and (20 <= abs(ang_im) <= 60)
            ):
                return "K"

        return None
