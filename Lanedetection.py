import cv2
import numpy as np


def multi_angle_edge_detection(image):
    g_0 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    g_45 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    g_90 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    g_135 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    edge_0 = cv2.filter2D(image, -1, g_0)  # Detects horizontal edges
    edge_45 = cv2.filter2D(image, -1, g_45)  # Detects edges with a 45-degree slope
    edge_90 = cv2.filter2D(image, -1, g_90)  # Detects vertical edges
    edge_135 = cv2.filter2D(image, -1, g_135)  # Detects edges with a 135-degree slope
    edges = np.sqrt(edge_0**2 + edge_45**2 + edge_90**2 + edge_135**2)

    return edges


def draw_lines(img, lines, color=(0, 0, 255), thickness=40):
    height, width = img.shape[0], img.shape[1]
    left_x = []
    left_y = []
    right_x = []
    right_y = []
    mid_point = width / 2 
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if y2 - y1 == 0:
                slope = 1
            else:
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) > 0.5:
                    if slope > 0 and x1 > mid_point and x2 > mid_point:
                        right_x.extend([x1, x2])
                        right_y.extend([y1, y2])
                    elif slope < 0 and x1 < mid_point and x2 < mid_point:
                        left_x.extend([x1, x2])
                        left_y.extend([y1, y2])
    
    # If there are suitable coordinates to create the left line
    if left_x and left_y and len(left_x) > 1 and len(left_y) > 1:
        # Calculating the coefficents of left line
        left_line_coef = np.polyfit(left_y, left_x, 1)
        l = np.poly1d(left_line_coef)

        min_height = int((1/5) * height)
        bottom_left_x = int(l(height))
        upper_left_x = int(l(min_height))
        cv2.line(img, (bottom_left_x, height), (upper_left_x, min_height), color, thickness)

    # If there are suitable coordinates to create the right line
    if right_x and right_y and len(right_x) > 1 and len(right_y) > 1:
        # Calculating the coefficents of right line
        right_line_coef = np.polyfit(right_y, right_x, 1)
        r = np.poly1d(right_line_coef)

        min_height = int((1/5) * height)
        bottom_right_x = int(r(height))
        upper_right_x = int(r(min_height))
        cv2.line(img, (bottom_right_x, height), (upper_right_x, min_height), color, thickness)

    try:
        bottom_left_x = int(l(height))
        upper_left_x = int(l(min_height))
        bottom_right_x = int(r(height))
        upper_right_x = int(r(min_height))
        cv2.rectangle(img, (bottom_left_x, min_height), (bottom_right_x, height), (255, 255, 0), -1)
    except:
        pass

  

def detect_direction(frame, edges):
    filtered_edges = cv2.filter2D(edges, -1, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32)) # Convolution with edge detection kernel
    y_min = 0

    A = np.column_stack((filtered_edges.flatten(), np.ones_like(filtered_edges.flatten())))
    coefficients, _, _, _ = np.linalg.lstsq(A, edges.flatten(), rcond=None)

    # Calculating the average y value for each column
    y_means = []
    x_values = []
    for x in range(edges.shape[1]):
        y = coefficients[0] * filtered_edges[:, x] + coefficients[1] + y_min
        y_mean = np.mean(y)
        y_means.append(y_mean)
        x_values.append(x)

    slope = (y_means[0] - y_means[-1]) / (x_values[0] - x_values[-1])
    if abs(slope) < 0.1:
        direction = "Straight"
    elif slope >0:
        direction = "Right"
    else:
        direction = "Left"

    cv2.putText(frame, direction, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame


cap = cv2.VideoCapture("solidWhiteRight.mp4")

while cap.isOpened():
    ret, img = cap.read()

    if not ret or img is None:
        break

    frame = cv2.resize(img, (640, 480))
    frame2 = frame

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    normalize = gray_frame / 255.0
    nonlinear_transform_img = cv2.pow(normalize, 4)

    img_edges = multi_angle_edge_detection(nonlinear_transform_img)
    img_edges = cv2.normalize(img_edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    tl = [220, 340]
    bl = [100, 440]
    tr = [440, 340]
    br = [580, 440]
    roi = [tl, bl, tr, br]

    cv2.line(frame2, tl, tr, (0, 0, 255), 1)
    cv2.line(frame2, tr, br, (0, 0, 255), 1)
    cv2.line(frame2, br, bl, (0, 0, 255), 1)
    cv2.line(frame2, bl, tl, (0, 0, 255), 1)

    # Parameters for Perspective Transform
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

    # Applying Perspective Transform on roi. 
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))
    edges = cv2.warpPerspective(img_edges, matrix, (640, 480))

    lines = cv2.HoughLinesP(edges, rho= 1, theta= np.pi/ 180, threshold=150, minLineLength=75, maxLineGap=3)
    draw_lines(transformed_frame, lines, color=(255, 0, 255))
    detect_direction(frame, edges)
    
    transformed_frame = cv2.pow(transformed_frame, -1)

    # Applying Inverse Perspective Transform 
    matrix2 = cv2.getPerspectiveTransform(pts2, pts1)
    reverse_transformed_frame = cv2.warpPerspective(transformed_frame, matrix2, (640, 480))
    result = cv2.addWeighted(frame2, 1, reverse_transformed_frame, 1, 0)

    # Results
    cv2.imshow("frame",frame)
    cv2.imshow("result",result)
    #cv2.imshow("nonlinear",nonlinear_transform_img)
    #cv2.imshow("multi angle edge detection", img_edges)
    #cv2.imshow("transformed frame", transformed_frame)
    #cv2.imshow("edges",edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
