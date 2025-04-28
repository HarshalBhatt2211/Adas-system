import numpy as np
import cv2
import math

class CalcAngle:
    def __init__(self,frame, lower_blue, upper_blue):
        self.frame = frame
        self.lower_blue = lower_blue
        self.upper_blue = upper_blue
        self.boundry = 1/3
        self.height, self.width, _ = frame.shape
        self.mid = int(self.width / 2)
        self.left_region_boundry = self.width * (1 - self.boundry)
        self.right_region_boundry = self.width * self.boundry

    def detect_edges(self):
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        edges = cv2.Canny(mask, 100, 200)
        return edges

    def region_of_interest(self, edges):
        mask = np.zeros_like(edges)

        polygon = np.array([[
            (0, self.height * 1/2),  # Top-left point
            (self.width, self.height * 1/2), # Top-right point
            (self.width, self.height), # Bottom-right point
            (0, self.height), # Bottom-left point
        ]], np.int32)

        cv2.fillPoly(mask, polygon, 255)

        masked_edges = cv2.bitwise_and(edges, mask)
        return masked_edges

    def detect_line_segments(self, masked_edges):
        rho = 1               # distance precision in pixel
        theta = np.pi / 180   # angular precision in radian
        min_threshold = 10    # minimum number of votes
        line_segments = cv2.HoughLinesP(
            masked_edges, rho, theta, min_threshold,
            np.array([]), minLineLength=20, maxLineGap=14
        )
        return line_segments
    def display_lines(self, line_segments):
        line_image = np.zeros_like(self.frame)
        if line_segments is not None:
            for line in line_segments:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        line_image = cv2.addWeighted(self.frame, 0.8, line_image, 1, 1)
        return line_image

    def make_points(self, fit_average):
        slope, intercept = fit_average
        y1 = self.height
        y2 = int(y1 * 1/2)

        # calculate x1 from x = (y-b)/m
        x1 = max(-self.width, min(2 * self.width, int((y1 - intercept) / slope)))
        x2 = max(-self.width, min(2 * self.width, int((y2 - intercept) / slope)))

        return [[x1, y1, x2, y2]]
    def average_slope_intercept(self, line_segments):
        left = []       # left lines
        right = []      # right lines
        lines = []      # all final averaged lines

        if line_segments is None:
            return -1
        for line in line_segments:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue
                # y = mx + b
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope = fit[0]
                intercept = fit[1]

                if slope < 0:
                    if x1 < self.left_region_boundry and x2 < self.left_region_boundry:
                        left.append((slope, intercept))
                else:
                    if x1 > self.right_region_boundry and x2 > self.right_region_boundry:
                        right.append((slope, intercept))

        left_avg = np.average(left, axis=0) if len(left) > 0 else None
        right_avg = np.average(right, axis=0) if len(right) > 0 else None

        if left_avg is not None:
            lines.append(self.make_points(left_avg))
        if right_avg is not None:
            lines.append(self.make_points(right_avg))
        return lines
    
    def compute_steering_angle(self,lines):
        if lines == -1 or len(lines) == 0:
            return 90
        
        if len(lines) == 1:
            x1, _, x2, _ = lines[0][0]
            x_offset = x2-x1
        else:

            _, _, left_x2, _ = lines[0][0]
            _, _, right_x2, _ = lines[1][0]

            x_offset = (left_x2 + right_x2) / 2 -self.mid
        
        y_offset = int(self.height / 2)
        angle_to_mid_deg = int(math.atan(x_offset / y_offset) * 180.0 / math.pi)
        return 90 - angle_to_mid_deg
    
    def get_angle(self):
        edges = self.detect_edges()
        masked_edges = self.region_of_interest(edges)
        line_segments = self.detect_line_segments(masked_edges)
        lines = self.average_slope_intercept(line_segments)
        angle = self.compute_steering_angle(lines)
        return angle
    
    if __name__ == '__main__':
        frame = cv2.imread('img1.jpg')
        lower_blue = np.array([80, 70, 40])
        upper_blue = np.array([180, 255, 255])
        angle = CalcAngle(frame, lower_blue, upper_blue)
        print(angle)
