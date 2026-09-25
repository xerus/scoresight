import cv2
import numpy as np


# This class is used to stabilize the frames of the video.
# It uses ORB features to match keypoints between frames and calculate an affine transform to
# warp the frame.
class FrameStabilizer:
    def __init__(self):
        self.stabilizationFrame = None
        self.stabilizationFrameCount = 0
        self.stabilizationBurnInCompleted = False
        self.stabilizationKPs = None
        self.stabilizationDesc = None
        self.orb = None
        self.matcher = None

    def reset(self):
        self.stabilizationFrame = None
        self.stabilizationFrameCount = 0
        self.stabilizationBurnInCompleted = False
        self.stabilizationKPs = None
        self.stabilizationDesc = None

    def stabilize_frame(self, frame_rgb):
        if self.stabilizationFrame is None:
            self.stabilizationFrame = frame_rgb.copy()
            self.stabilizationFrameCount = 0
        elif frame_rgb.shape != self.stabilizationFrame.shape:
            # Video sources can change resolution after a reload. Rebuild the
            # reference instead of comparing descriptors from different sizes.
            self.reset()
            self.stabilizationFrame = frame_rgb.copy()
        elif not self.stabilizationBurnInCompleted:
            self.stabilizationFrameCount += 1
            # add the new frame to the stabilization frame
            frame_rgb = cv2.addWeighted(frame_rgb, 0.5, self.stabilizationFrame, 0.5, 0)
            if self.stabilizationFrameCount == 10:
                self.stabilizationBurnInCompleted = True
                # extract ORB features from the stabilization frame
                self.orb = cv2.ORB_create()
                self.stabilizationKPs, self.stabilizationDesc = (
                    self.orb.detectAndCompute(self.stabilizationFrame, None)
                )
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        if (
            self.stabilizationBurnInCompleted
            and self.stabilizationFrame is not None
            and self.orb is not None
            and self.matcher is not None
            and self.stabilizationKPs is not None
            and self.stabilizationDesc is not None
        ):
            try:
                # Feature detectors can return no descriptors for blank, dark,
                # or heavily blurred frames. In that case, show the source frame
                # unchanged and try again on the next frame.
                kps, desc = self.orb.detectAndCompute(frame_rgb, None)
                if kps is None or desc is None or len(kps) < 6:
                    return frame_rgb

                matches = self.matcher.match(self.stabilizationDesc, desc)
                if matches is None or len(matches) < 6:
                    return frame_rgb

                matches = sorted(matches, key=lambda match: match.distance)[:100]
                src_pts = np.float32(
                    [self.stabilizationKPs[match.queryIdx].pt for match in matches]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [kps[match.trainIdx].pt for match in matches]
                ).reshape(-1, 1, 2)
                if src_pts.shape != dst_pts.shape or len(src_pts) < 6:
                    return frame_rgb

                transform, inliers = cv2.estimateAffinePartial2D(
                    src_pts,
                    dst_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                    maxIters=2000,
                    confidence=0.99,
                    refineIters=10,
                )
            except cv2.error:
                # A bad/degenerate set of ORB matches must not terminate the
                # camera thread. Continue with the unmodified frame instead.
                return frame_rgb

            if transform is None or inliers is None:
                return frame_rgb
            if not np.isfinite(transform).all():
                return frame_rgb

            inlier_count = int(inliers.sum())
            if inlier_count < 4 or inlier_count / len(matches) < 0.25:
                return frame_rgb

            scale = np.hypot(transform[0, 0], transform[1, 0])
            angle = np.degrees(np.arctan2(transform[1, 0], transform[0, 0]))
            height, width = frame_rgb.shape[:2]
            tx, ty = transform[:, 2]
            if (
                not 0.8 <= scale <= 1.25
                or abs(angle) > 20
                or abs(tx) > width * 0.2
                or abs(ty) > height * 0.2
            ):
                return frame_rgb

            try:
                frame_rgb = cv2.warpAffine(
                    frame_rgb,
                    transform,
                    (width, height),
                    flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                )
            except cv2.error:
                # Stabilization is optional; a warp failure should preserve the
                # current frame rather than stop video processing.
                pass

        return frame_rgb
