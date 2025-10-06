import cv2
import numpy as np

points = []

class Projection(object):

    def __init__(self, image_path, points,
                 bev_orientation_deg=(-90, 0.0, 0.0),
                 bev_position=(0.0, 0.0, 0.0),
                 front_orientation_deg=(0.0, 0.0, 0.0),
                 front_position=(0.0, 1.0, 0.0),
                 fov=90.0):
        """
            :param points: Selected pixels on top view(BEV) image
            :param bev_orientation_deg: (pitch, yaw, roll) of BEV camera in degrees
            :param bev_position: (x, y, z) of BEV camera centre in world coordinates
            :param front_orientation_deg: (pitch, yaw, roll) of front camera in degrees
            :param front_position: (x, y, z) of front camera centre in world coordinates
            :param fov: Horizontal field of view (degrees) shared by both cameras
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape

        self.src_pts = np.asarray(points, dtype=np.float64) # selected points in top view
        if self.src_pts.ndim != 2 or self.src_pts.shape[1] != 2 or len(self.src_pts) < 3:
            raise ValueError("Need at least 3 clicked points (Nx2) from BEV image.")

        self.fov = float(fov)

        self.bev_orientation = self._as_radians(bev_orientation_deg, 'BEV orientation')
        self.front_orientation = self._as_radians(front_orientation_deg, 'front orientation')

        self.bev_position = self._as_vector(bev_position, 'BEV position')
        self.front_position = self._as_vector(front_position, 'front position')

    def _as_radians(self, values, name):
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"Expected {name} to be a 3-element iterable.")
        return np.deg2rad(arr)

    def _as_vector(self, values, name):
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"Expected {name} to be a 3-element iterable.")
        return arr
    
    def intrinsics_from_fov(self, W, H, fov_deg):
        """ Calculate the intrinsic parameters of a camera """
        f = 0.5 * W / np.tan(np.deg2rad(fov_deg) / 2.0)
        cx, cy = W * 0.5, H * 0.5
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0,  1]], dtype=np.float64)
        return K
    
    def Rx(self, a):
        ca, sa = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0],
                         [0, ca, -sa],
                         [0, sa,  ca]], dtype=np.float64)

    def Ry(self, a):
        ca, sa = np.cos(a), np.sin(a)
        return np.array([[ ca, 0, sa],
                         [  0, 1,  0],
                         [-sa, 0, ca]], dtype=np.float64)

    def Rz(self, a):
        ca, sa = np.cos(a), np.sin(a)
        return np.array([[ca, -sa, 0],
                         [sa,  ca, 0],
                         [ 0,   0, 1]], dtype=np.float64)

    # Compose as R_c2w = Rz(roll) * Ry(yaw) * Rx(pitch)
    def R_c2w_from_euler(self, pitch, yaw, roll):
        return self.Rz(roll) @ self.Ry(yaw) @ self.Rx(pitch)

    def top_to_front(self):
        """
        Project the selected BEV pixels onto the front view using
        the relative pose between the two cameras.
        """

        H, W = self.height, self.width
        K = self.intrinsics_from_fov(W, H, self.fov)
        Kinv = np.linalg.inv(K)

        # Front view (target) camera pose.
        C_front = self.front_position
        R_front_c2w = self.R_c2w_from_euler(*self.front_orientation)
        R_front_w2c = R_front_c2w.T

        # BEV (source) camera pose derived from constructor inputs.
        R_bev_c2w = self.R_c2w_from_euler(*self.bev_orientation)
        C_bev = self.bev_position

        new_pixels = []
        eps = 1e-8
        for u, v in self.src_pts:
            # Convert pixel to a ray in the BEV camera frame.
            ray_bev = Kinv @ np.array([u, v, 1.0])
            ray_world = R_bev_c2w @ ray_bev

            if abs(ray_world[1]) < eps:
                continue  # Ray parallel to the ground plane.

            # Intersect with the ground plane (Y_world = 0).
            lam = -C_bev[1] / ray_world[1]

            X_world = C_bev + lam * ray_world
            X_front = R_front_w2c @ (X_world - C_front)
            x_h = K @ X_front
            if abs(x_h[2]) < eps:
                print('The projected point is out of range !!')
                continue  # Avoid division by zero
            pixel = np.round(x_h[:2] / x_h[2]).astype(int)
            pixel = np.clip(pixel, 0, W)
            new_pixels.append(pixel.tolist())

        return new_pixels

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """

        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


if __name__ == "__main__":

    pitch_ang = -90

    front_rgb = "bev_data/front2.png"
    top_rgb = "bev_data/bev2.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    bev_orientation = (pitch_ang, 0.0, 0.0)
    bev_position = (0.0, 2.5, 0.0)
    front_orientation = (0.0, 0.0, 0.0)
    front_position = (0.0, 1.0, 0.0)

    projection = Projection(
        front_rgb,
        points,
        bev_orientation_deg=bev_orientation,
        bev_position=bev_position,
        front_orientation_deg=front_orientation,
        front_position=front_position)

    new_pixels = projection.top_to_front()
    projection.show_image(new_pixels)
