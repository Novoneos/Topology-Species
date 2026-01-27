import numpy as np
from scipy.special import binom
from PIL import Image
import cv2
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import io, cv2

class Shape2D:
    def __init__(self, dim=64):
        """
        Initialize the base Shape2D class.
        
        This is the parent class for all 2D shape generators. It provides common
        functionality including image storage, visualization, rotation, and export
        capabilities that all shape subclasses inherit.
        
        Parameters:
        - dim: int, dimension of the output binary image (creates dim x dim array)
               Default is 64 for 64x64 pixel resolution
        
        Attributes:
        - DIM: int, stores the image dimension
        - img: numpy.ndarray, boolean array storing the final binary shape
        - rotation_angle: float, tracks cumulative rotation applied to shape
        - color0: list, RGB color for False/0 values in visualization (default white)
        - color1: list, RGB color for True/1 values in visualization (default blue-gray)
        - cmap: matplotlib colormap for visualization
        """
        self.DIM = dim
        self.img = np.zeros((self.DIM, self.DIM), dtype=bool)
        self.rotation_angle = 0
        self.color0 = [0.13, 0.13, 0.13]  # White background
        self.color1 = [0.18, 0.53, 0.83]  # Blue-gray for shape
        self.cmap = ListedColormap([self.color0, self.color1])

    def flip_quadrants(self, arr):
        """
        Flips the top left quadrant of the input array horizontally and vertically,
        and copies the result into the other three quadrants to produce a 2n x 2n array.
        """
        n = arr.shape[0]
        # create an empty 2n x 2n numpy array
        new_arr = np.zeros((2*n, 2*n), dtype=bool)
        # copy the original array to the top left quadrant
        new_arr[:n, :n] = arr
        # flip the top left quadrant horizontally and copy it to the top right quadrant
        new_arr[:n, n:] = np.fliplr(arr)
        # flip the top left quadrant vertically and copy it to the bottom left quadrant
        new_arr[n:, :n] = np.flipud(arr)
        # flip the top left quadrant both horizontally and vertically and copy it to the bottom right quadrant
        new_arr[n:, n:] = np.flip(arr)
        return new_arr

    def draw_shape(self, color0=None, color1=None, savefig=False, name='shape'):
        """
        Plot a 2D binary numpy array with specified colors for 0 and 1.
    
        Parameters:
            array (ndarray): 2D binary array.
            color0 (list): RGB color for the value 0. Default is black.
            color1 (list): RGB color for the value 1. Default is white.
        """
        
        if color0 is not None and color1 is not None:
            self.color0 = color0
            self.color1 = color1
            self.cmap = ListedColormap([self.color0, self.color1])
        
        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(self.img, cmap=self.cmap, interpolation='nearest', origin='upper')
        ax.axis('off')
        if savefig:
            fig.savefig(name + '.png', bbox_inches='tight', pad_inches=0)
            plt.clf()
        else: plt.show()
        
    def savebinary(self, p=0.3, h=0.4, file_path='binaryshape.txt'):
        """
        Save a 2D array (meta atom) to a text file in a specified format.
    
        Parameters:
        - img: 2D numpy array representing the meta atom
        - p: period of the meta atom
        - h: height of the meta atom
        - file_path: path to save the txt file
        """
        n = self.img.shape[0]  # Assuming img is a square array
        header = [
            f"{n} {-p/2} {p/2}",
            f"{n} {-p/2} {p/2}",
            f"2 {-h/2} {h/2}"
        ]
    
        # Flatten the 2D img array to 1D
        flat_img = self.img.flatten()
    
        # Add n^2 zeros
        zeros = np.zeros(n ** 2, dtype=int)
    
        # Combine header, flat_img, and zeros
        content = "\n".join(header) + "\n" + "\n".join(map(str, flat_img)) + "\n" + "\n".join(map(str, zeros))
    
        # Save to file
        with open(file_path, 'w') as f:
            f.write(content)
            
    def rotate(self, angle):
        """
        Rotate the given boolean array 'img' by 'angle' degrees.
        
        Parameters:
        - angle: float, rotation angle in degrees, counter-clockwise.
        
        Returns:
        - None
        """
        # Rotate the image. 'reshape=False' keeps the output array's shape the same as the input.
        # 'order=0' uses nearest-neighbor interpolation, 'mode='constant'', 'cval=0.0' fills the
        # background with False (0) after rotation.
        rotated_img = rotate(self.img.astype(float), angle, reshape=False, order=0, mode='constant', cval=0.0)
        self.img = rotated_img > 0.5
        self.rotation_angle += angle


class Segment:
    """
    This class defines a segment of a Bezier curve with specified start and end points and angles.
    """
    def __init__(self, p1, p2, angle1, angle2, numpoints=100, r=0.3):
        """
        Initialize a Bezier curve segment.

        Parameters:
        - p1: numpy array with the starting point of the segment.
        - p2: numpy array with the ending point of the segment.
        - angle1: angle at which the curve starts.
        - angle2: angle at which the curve ends.
        - numpoints: number of points to generate for this segment.
        - r: scaling factor for the control points distance.
        """
        self.p1 = p1
        self.p2 = p2
        self.angle1 = angle1
        self.angle2 = angle2
        self.numpoints = numpoints
        self.r = r * np.sqrt(np.sum((self.p2 - self.p1) ** 2))
        self.p = np.zeros((4, 2))
        self.curve = None  # This attribute will store the generated Bezier curve
        self.initialize_control_points()
        self.generate_curve()  # Generating the curve during initialization

    def initialize_control_points(self):
        """
        Initialize the control points for the Bezier segment based on the start and end points and angles.
        """
        self.p[0] = self.p1
        self.p[3] = self.p2
        self.p[1] = self.p1 + np.array([np.cos(self.angle1), np.sin(self.angle1)]) * self.r
        self.p[2] = self.p2 - np.array([np.cos(self.angle2), np.sin(self.angle2)]) * self.r

    def bernstein(self, n, k, t):
        """
        Calculate the Bernstein polynomial for given n, k, and t.
        
        Parameters:
        - n: integer, the degree of the Bernstein polynomial.
        - k: integer, the index of the Bernstein basis function.
        - t: float, parameter along the curve (0 <= t <= 1).

        Returns:
        - float, the value of the Bernstein polynomial at parameter t.
        """
        return binom(n, k) * t ** k * (1 - t) ** (n - k)

    def generate_curve(self):
        """
        Generate and store the Bezier curve in the 'curve' attribute.
        """
        x_vals, y_vals = np.array([0.0] * self.numpoints), np.array([0.0] * self.numpoints)
        n = len(self.p) - 1
        for i in range(n + 1):
            x_vals += self.bernstein(n, i, np.linspace(0.0, 1.0, self.numpoints)) * self.p[i, 0]
            y_vals += self.bernstein(n, i, np.linspace(0.0, 1.0, self.numpoints)) * self.p[i, 1]
        self.curve = np.column_stack((x_vals, y_vals))


class Bezier(Shape2D):
    """
    This class defines a Bezier shape composed of multiple Bezier segments.
    """
    def __init__(self, points=None, n_anchor_points=None, dim=64, rad=0.5, edgy=0.05, segpoints=25, sym=False):
        """
        Initialize BezierShape instance. Either points or n_bezier_points should be provided.

        :param points: A predefined set of control points for the Bezier curve.
        :param n_bezier_points: Number of control points to generate a random Bezier curve.
        :param dim: Size of the binary image.
        :param rad: Radius for Bezier curve generation.
        :param edgy: Factor for edginess in Bezier curve.
        :param segpoints: Number of points for curve segmentation.
        """
        super().__init__(dim)
        self.points = points
        self.n_anchor_points = n_anchor_points
        self.x, self.y = None, None
        self.RAD = rad  # Radius for Bezier curve generation
        self.EDGY = edgy  # Factor for edgy-ness in Bezier curve
        self.SEGMENTLENGTH = segpoints
        self.SYM = sym
        
        # points were given -> generate bezier shape from predefined points
        if self.points is None and self.n_anchor_points is None:
            self.n_anchor_points = 5
            
        if self.points is not None and self.n_anchor_points is None:
            self.n_anchor_points = len(self.points)
            self.generate_shape_from_points()
        # number of bezier points were given -> generate new bezier shape
        elif self.n_anchor_points is not None and self.points is None:
            self.generate_random_points()
            self.generate_shape_from_points()
        # something went wrong
        else:
            raise ValueError("Either points or n_anchor_points should be provided.")
    
    def bernstein(self, n, k, t):
        """
        Calculate the Bernstein polynomial for given n, k, and t.

        :param n: Degree of the polynomial.
        :param k: Current term.
        :param t: Parameter value.
        :return: The value of the Bernstein polynomial.
        """
        return binom(n, k) * t ** k * (1 - t) ** (n - k)
    
    def ccw_sort(self, p):
        """
        Sort points counter-clockwise around their centroid.

        :param p: 2D array containing the points to sort.
        :return: The points sorted in counter-clockwise order around their centroid.
        """
        d = p - np.mean(p, axis=0)
        s = np.arctan2(d[:, 0], d[:, 1])
        return p[np.argsort(s), :]
    
    def generate_random_points(self, scale=0.8, mindst=None, rec=0):
        """
        Generate n random points within the unit square.

        :param scale: Scaling factor for the points.
        :param mindst: Minimum distance between points.
        :param rec: Recursion count for generating points.
        """
        # Generate n random points within the unit square
        mindst = mindst or 0.7 / self.n_anchor_points
        # Set minimum distance between points, default to 0.7/n if not specified
        a = np.random.rand(self.n_anchor_points, 2)
        # Create n random points in a 2D array
        d = np.sqrt(np.sum(np.diff(self.ccw_sort(a), axis=0), axis=1) ** 2)
        # Calculate the distance between each point
        if np.all(d >= mindst) or rec >= 200:
            self.points = a * scale
            return
        else:
            return self.generate_random_points(scale=scale, mindst=mindst, rec=rec + 1)
        
    # This function generates a curve defined by a set of points and returns its segments and curve as NumPy arrays
    def get_curve(self, points):
        """
        Generate a curve defined by a set of points and return its segments and curve as NumPy arrays.

        :param points: A NumPy array of shape (n, 3) containing n points with x, y, and curvature values.
        :return: A tuple of two NumPy arrays:
                    - segments: A list of n-1 Segment objects, each representing a segment between two adjacent points.
                    - curve: A NumPy array of shape (m, 2) representing the curve obtained by concatenating all the segments.
        """
        segments = []
        for i in range(len(points) - 1):
            # Create a Segment object between each pair of points
            seg = Segment(points[i, :2],
                          points[i + 1, :2],
                          points[i, 2],
                          points[i + 1, 2],
                          self.SEGMENTLENGTH,
                          self.RAD)
            segments.append(seg)
        # Concatenate the curve of each segment to obtain the full curve
        curve = np.concatenate([s.curve for s in segments])
        return segments, curve
        
    def generate_bezier_curve(self):
        """
        Given an array of points *a*, create a curve through those points. 
        *rad* is a number between 0 and 1 to steer the distance of control points.
        *edgy* is a parameter which controls how "edgy" the curve is,
        edgy=0 is smoothest.
        """
        # Calculate points on a Bezier curve given an array of points
        p = np.arctan(self.EDGY) / np.pi + 0.5
        # Convert the edginess parameter to an angle in radians
        a = self.ccw_sort(self.points)
        # Sort points in a counterclockwise order
        a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
        # Append first point to the end for smooth curve creation
        d = np.diff(a, axis=0)
        # Calculate differences between points in the array
        ang = np.arctan2(d[:, 1], d[:, 0])
        # Calculate angle of each point from the x-axis
        ang1 = ang
        ang2 = np.roll(ang, 1)
        ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
        # Calculate new angle based on edginess and if the angle change is greater than pi radians
        ang = np.append(ang, [ang[0]])
        # Append the first angle to the end for smooth curve creation
        a = np.append(a, np.atleast_2d(ang).T, axis=1)
        # Append the new angles to the original array
        _, c = self.get_curve(a)
        # Get a smoothed curve from the array of points
        self.x, self.y = c.T
        # Separate the x and y coordinates of the curve

    def draw_shape_to_array(self, padding=0.2, threshold=0.5):
        """
        Convert the Bezier curve to a binary image.
        """

        # Create a 1-panel plot and fill it with white color
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_facecolor("black")
        ax.fill(self.x, self.y, zorder=1, color="white")
        
        # Set the limits
        xmin, xmax = self.x.min(), self.x.max()
        ymin, ymax = self.y.min(), self.y.max()
        
        x_pad_min = xmin - padding
        x_pad_max = xmax + padding
        y_pad_min = ymin - padding
        y_pad_max = ymax + padding
        
        if xmin >= 0 and ymin >= 0 and xmax <= 1 and ymax <= 1:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            ax.set_xlim(x_pad_min, x_pad_max)
            ax.set_ylim(y_pad_min, y_pad_max)
        
        # border = abs(min([shape.x.min(), shape.y.min()]))-padding
        # plt.xlim(0 - border, 1 + border)
        # plt.ylim(0 - border, 1 + border)
        
        # Turn off the axis, and add a background patch
        ax.set_axis_off()
        ax.add_artist(ax.patch)
        ax.patch.set_zorder(-1)
    
        # Save the plot as a PNG image in memory and read it as a NumPy array
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        im = Image.open(buf)
        img = np.array(im)
        buf.close()
        plt.close(fig)
        
        if self.SYM:
            # Resize the image and convert it to grayscale
            img_resized = cv2.resize(img, (self.DIM // 2, self.DIM // 2))
            img_grey = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
            # Thresholding the grayscale image to convert it to binary
            _, img_binary = cv2.threshold(img_grey, threshold, 1, cv2.THRESH_BINARY)
            # flip image for symmetry
            self.img = self.flip_quadrants(img_binary)
        else:
            # Resize the image and convert it to grayscale
            img_resized = cv2.resize(img, (self.DIM, self.DIM))
            img_grey = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
            # Thresholding the grayscale image to convert it to binary
            _, img_binary = cv2.threshold(img_grey, threshold, 1, cv2.THRESH_BINARY)
            self.img = img_binary
    
    def generate_shape_from_points(self):
        """
        Generate Bezier shape from predefined control points.
        """
        self.generate_bezier_curve()
        self.draw_shape_to_array()
        
    def get_shape_data(self):
        """
        Return all the class variables.
        """
        return self.points, self.x, self.y, self.img


class BezierStar(Shape2D):
    def __init__(self, dim=64, points=None, n_anchor_points=None, radmin=0.1, radmax=0.49, rad=0.5, edgy=0.05, segpoints=25):
        
        super().__init__(dim)  # Call the parent class constructor
        self.points = points
        self.x, self.y = None, None
        self.N_ANCHOR_POINTS = n_anchor_points
        self.RADMIN = radmin
        self.RADMAX = radmax
        self.RAD = rad  # Radius for Bezier curve generation
        self.EDGY = edgy  # Factor for edgy-ness in Bezier curve
        self.SEGMENTLENGTH = segpoints
        
        # points were given -> generate bezier shape from predefined points
        if self.points is None and self.N_ANCHOR_POINTS is None:
            self.N_ANCHOR_POINTS = 5
            
        if self.points is not None and self.N_ANCHOR_POINTS is None:
            self.N_ANCHOR_POINTS = len(self.points)
            self.generate_shape_from_points()
        # number of bezier points were given -> generate new bezier shape
        elif self.N_ANCHOR_POINTS is not None and self.points is None:
            self.generate_random_points()
            self.generate_shape_from_points()
        # something went wrong
        else:
            raise ValueError("Either points or n_anchor_points should be provided.")
            
    def get_shape_data(self):
        return self.x, self.y, self.points, self.img
        
    def generate_random_points(self):
        """
        defines points around the center of the unit square with a variable distance.

        Returns
        -------
        None.

        """
        # Define the center of the unit square
        center_x, center_y = 0.5, 0.5
        
        # Calculate the angles for the anchor points
        angles = np.linspace(0, 90, self.N_ANCHOR_POINTS, endpoint=True)
        
        # Generate random distances from the center for each anchor point
        distances = np.random.uniform(self.RADMIN, self.RADMAX, size=self.N_ANCHOR_POINTS)
        distances[-1] = distances[0]
        
        # Calculate the coordinates for the anchor points in the first quadrant
        anchor_x_1st = center_x + distances * np.cos(np.radians(angles))
        anchor_y_1st = center_y + distances * np.sin(np.radians(angles))
        
        # Mirror the points to the other quadrants
        anchor_x = np.concatenate([anchor_x_1st, 1 - anchor_x_1st, 1 - anchor_x_1st, anchor_x_1st])
        anchor_y = np.concatenate([anchor_y_1st, anchor_y_1st, 1 - anchor_y_1st, 1 - anchor_y_1st])
        
        self.points = np.unique(np.column_stack((anchor_x, anchor_y)), axis=0)
        
        
    def generate_shape_from_points(self):
        """
        Generate Bezier shape from predefined control points.
        """
        self.generate_bezier_curve()
        self.draw_shape_to_array()
        
    # This function generates a curve defined by a set of points and returns its segments and curve as NumPy arrays
    def get_curve(self, points):
        """
        Generate a curve defined by a set of points and return its segments and curve as NumPy arrays.

        :param points: A NumPy array of shape (n, 3) containing n points with x, y, and curvature values.
        :return: A tuple of two NumPy arrays:
                    - segments: A list of n-1 Segment objects, each representing a segment between two adjacent points.
                    - curve: A NumPy array of shape (m, 2) representing the curve obtained by concatenating all the segments.
        """
        segments = []
        for i in range(len(points) - 1):
            # Create a Segment object between each pair of points
            seg = Segment(points[i, :2],
                          points[i + 1, :2],
                          points[i, 2],
                          points[i + 1, 2],
                          self.SEGMENTLENGTH,
                          self.RAD)
            segments.append(seg)
        # Concatenate the curve of each segment to obtain the full curve
        curve = np.concatenate([s.curve for s in segments])
        return segments, curve
    
    def bernstein(self, n, k, t):
        """
        Calculate the Bernstein polynomial for given n, k, and t.

        :param n: Degree of the polynomial.
        :param k: Current term.
        :param t: Parameter value.
        :return: The value of the Bernstein polynomial.
        """
        return binom(n, k) * t ** k * (1 - t) ** (n - k)
    
    def ccw_sort(self, p):
        """
        Sort points counter-clockwise around their centroid.

        :param p: 2D array containing the points to sort.
        :return: The points sorted in counter-clockwise order around their centroid.
        """
        d = p - np.mean(p, axis=0)
        s = np.arctan2(d[:, 0], d[:, 1])
        return p[np.argsort(s), :]
        
    def generate_bezier_curve(self):
        """
        Given an array of points *a*, create a curve through those points. 
        *rad* is a number between 0 and 1 to steer the distance of control points.
        *edgy* is a parameter which controls how "edgy" the curve is,
        edgy=0 is smoothest.
        """
        # Calculate points on a Bezier curve given an array of points
        p = np.arctan(self.EDGY) / np.pi + 0.5
        # Convert the edginess parameter to an angle in radians
        a = self.ccw_sort(self.points)
        # Sort points in a counterclockwise order
        a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
        # Append first point to the end for smooth curve creation
        d = np.diff(a, axis=0)
        # Calculate differences between points in the array
        ang = np.arctan2(d[:, 1], d[:, 0])
        # Calculate angle of each point from the x-axis
        ang1 = ang
        ang2 = np.roll(ang, 1)
        ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
        # Calculate new angle based on edginess and if the angle change is greater than pi radians
        ang = np.append(ang, [ang[0]])
        # Append the first angle to the end for smooth curve creation
        a = np.append(a, np.atleast_2d(ang).T, axis=1)
        # Append the new angles to the original array
        _, c = self.get_curve(a)
        # Get a smoothed curve from the array of points
        self.x, self.y = c.T
        # Separate the x and y coordinates of the curve
        
    def draw_shape_to_array(self):
        """
        Convert the Bezier curve to a binary image.
        """
        threshold = 0.5

        # Create a 1-panel plot and fill it with white color
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_facecolor("black")
        ax.fill(self.x, self.y, zorder=1, color="white")
        # Set the limits, turn off the axis, and add a background patch
        border = 0.2
        plt.xlim(0 - border, 1 + border)
        plt.ylim(0 - border, 1 + border)
        ax.set_axis_off()
        ax.add_artist(ax.patch)
        ax.patch.set_zorder(-1)
    
        # Save the plot as a PNG image in memory and read it as a NumPy array
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        im = Image.open(buf)
        img = np.array(im)
        buf.close()
        plt.close(fig)
        
        # Resize the image and convert it to grayscale
        img_resized = cv2.resize(img, (self.DIM, self.DIM))
        img_grey = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
        # Thresholding the grayscale image to convert it to binary
        _, img_binary = cv2.threshold(img_grey, threshold, 1, cv2.THRESH_BINARY)
        self.img = img_binary
        
        
class BezierFlower(Shape2D):
    def __init__(self, dim=64, points=None, n_anchor_points=None, radmin=0.1, radmax=0.49, rad=0.5, edgy=0.05, segpoints=25):
        
        super().__init__(dim)  # Call the parent class constructor
        self.points = points
        self.x, self.y = None, None
        self.N_ANCHOR_POINTS = n_anchor_points
        self.RADMIN = radmin
        self.RADMAX = radmax
        self.RAD = rad  # Radius for Bezier curve generation
        self.EDGY = edgy  # Factor for edgy-ness in Bezier curve
        self.SEGMENTLENGTH = segpoints
        
        # points were given -> generate bezier shape from predefined points
        if self.points is None and self.N_ANCHOR_POINTS is None:
            self.N_ANCHOR_POINTS = 3
            
        if self.points is not None and self.N_ANCHOR_POINTS is None:
            self.N_ANCHOR_POINTS = len(self.points)
            self.generate_shape_from_points()
        # number of bezier points were given -> generate new bezier shape
        elif self.N_ANCHOR_POINTS is not None and self.points is None:
            self.generate_random_points()
            self.generate_shape_from_points()
        # something went wrong
        else:
            raise ValueError("Either points or n_anchor_points should be provided.")
        
    def generate_random_points(self):
        """
        defines points around the center of the unit square with a variable distance.

        Returns
        -------
        None.

        """
        # Define the center of the unit square
        center_x, center_y = 0.5, 0.5
        
        # Calculate the angles for the anchor points
        angles = np.linspace(0, 45, self.N_ANCHOR_POINTS, endpoint=True)
        
        # Generate random distances from the center for each anchor point
        distances = np.random.uniform(self.RADMIN, self.RADMAX, size=self.N_ANCHOR_POINTS)
        distances[-1] = distances[0]
        
        # Calculate the coordinates for the anchor points in the first quadrant
        anchor_x_1st = center_x + distances * np.cos(np.radians(angles))
        anchor_y_1st = center_y + distances * np.sin(np.radians(angles))
    
        # # Mirror the points to the other quadrants and half-quadrants
        # anchor_x = np.concatenate([anchor_x_1st, 1 - anchor_x_1st, 1 - anchor_x_1st, anchor_x_1st,
        #                             anchor_y_1st, 1 - anchor_y_1st, 1 - anchor_y_1st, anchor_y_1st])
        # anchor_y = np.concatenate([anchor_y_1st, anchor_y_1st, 1 - anchor_y_1st, 1 - anchor_y_1st,
        #                             anchor_x_1st, anchor_x_1st, 1 - anchor_x_1st, 1 - anchor_x_1st])
        
        anchor_x_q = np.concatenate([anchor_x_1st, anchor_y_1st[::-1]])
        anchor_y_q = np.concatenate([anchor_y_1st, anchor_x_1st[::-1]])
        
        # Mirror the points to the other quadrants
        anchor_x = np.concatenate([anchor_x_q, 1 - anchor_x_q, 1 - anchor_x_q, anchor_x_q])
        anchor_y = np.concatenate([anchor_y_q, anchor_y_q, 1 - anchor_y_q, 1 - anchor_y_q])
        
        self.points = np.unique(np.column_stack((anchor_x, anchor_y)), axis=0)
        
    def generate_shape_from_points(self):
        """
        Generate Bezier shape from predefined control points.
        """
        self.generate_bezier_curve()
        self.draw_shape_to_array()
        
    # This function generates a curve defined by a set of points and returns its segments and curve as NumPy arrays
    def get_curve(self, points):
        """
        Generate a curve defined by a set of points and return its segments and curve as NumPy arrays.

        :param points: A NumPy array of shape (n, 3) containing n points with x, y, and curvature values.
        :return: A tuple of two NumPy arrays:
                    - segments: A list of n-1 Segment objects, each representing a segment between two adjacent points.
                    - curve: A NumPy array of shape (m, 2) representing the curve obtained by concatenating all the segments.
        """
        segments = []
        for i in range(len(points) - 1):
            # Create a Segment object between each pair of points
            seg = Segment(points[i, :2],
                          points[i + 1, :2],
                          points[i, 2],
                          points[i + 1, 2],
                          self.SEGMENTLENGTH,
                          self.RAD)
            segments.append(seg)
        # Concatenate the curve of each segment to obtain the full curve
        curve = np.concatenate([s.curve for s in segments])
        return segments, curve
    
    def bernstein(self, n, k, t):
        """
        Calculate the Bernstein polynomial for given n, k, and t.

        :param n: Degree of the polynomial.
        :param k: Current term.
        :param t: Parameter value.
        :return: The value of the Bernstein polynomial.
        """
        return binom(n, k) * t ** k * (1 - t) ** (n - k)
    
    def ccw_sort(self, p):
        """
        Sort points counter-clockwise around their centroid.

        :param p: 2D array containing the points to sort.
        :return: The points sorted in counter-clockwise order around their centroid.
        """
        d = p - np.mean(p, axis=0)
        s = np.arctan2(d[:, 0], d[:, 1])
        return p[np.argsort(s), :]
        
    def generate_bezier_curve(self):
        """
        Given an array of points *a*, create a curve through those points. 
        *rad* is a number between 0 and 1 to steer the distance of control points.
        *edgy* is a parameter which controls how "edgy" the curve is,
        edgy=0 is smoothest.
        """
        # Calculate points on a Bezier curve given an array of points
        p = np.arctan(self.EDGY) / np.pi + 0.5
        # Convert the edginess parameter to an angle in radians
        a = self.ccw_sort(self.points)
        # Sort points in a counterclockwise order
        a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
        # Append first point to the end for smooth curve creation
        d = np.diff(a, axis=0)
        # Calculate differences between points in the array
        ang = np.arctan2(d[:, 1], d[:, 0])
        # Calculate angle of each point from the x-axis
        ang1 = ang
        ang2 = np.roll(ang, 1)
        ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
        # Calculate new angle based on edginess and if the angle change is greater than pi radians
        ang = np.append(ang, [ang[0]])
        # Append the first angle to the end for smooth curve creation
        a = np.append(a, np.atleast_2d(ang).T, axis=1)
        # Append the new angles to the original array
        _, c = self.get_curve(a)
        # Get a smoothed curve from the array of points
        self.x, self.y = c.T
        # Separate the x and y coordinates of the curve
        
    def draw_shape_to_array(self):
        """
        Convert the Bezier curve to a binary image.
        """
        threshold = 0.5

        # Create a 1-panel plot and fill it with white color
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_facecolor("black")
        ax.fill(self.x, self.y, zorder=1, color="white")
        # Set the limits, turn off the axis, and add a background patch
        border = 0.2
        plt.xlim(0 - border, 1 + border)
        plt.ylim(0 - border, 1 + border)
        ax.set_axis_off()
        ax.add_artist(ax.patch)
        ax.patch.set_zorder(-1)
    
        # Save the plot as a PNG image in memory and read it as a NumPy array
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        im = Image.open(buf)
        img = np.array(im)
        buf.close()
        plt.close(fig)
        
        # Resize the image and convert it to grayscale
        img_resized = cv2.resize(img, (self.DIM, self.DIM))
        img_grey = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
        # Thresholding the grayscale image to convert it to binary
        _, img_binary = cv2.threshold(img_grey, threshold, 1, cv2.THRESH_BINARY)
        self.img = img_binary


class NeedleDrop(Shape2D):
    """
    The NeedleDrop class is a subclass of Shape2D that initializes a 2D image
    consisting of quadrants with flipped rectangles. 
    """

    def __init__(self, dim=64, limits=None, num_drops=3, margin=3, minsize=3):
        """
        Initializes the NeedleDrop object.
        
        Parameters:
            dim (int): Dimension of the 2D shape.
            num_drops (int): Number of rectangles to drop.
            margin (int): Margin from the edges of the shape.
            minsize (int): Minimum size of the rectangle.
        """
        super().__init__(dim)  # Call the parent class constructor
        self.NUM_DROPS = num_drops
        self.MARGIN = margin
        self.MINSIZE = minsize
        self.limits = limits
        
        if self.limits is None:
            self.quadrant = self.get_random_quadrant()  # Create the first quadrant
            self.img = self.flip_quadrants(self.quadrant)  # Create the full image by flipping quadrants
        else:
            self.quadrant = self.get_quadrant()  # Create the first quadrant
            self.img = self.flip_quadrants(self.quadrant)  # Create the full image by flipping quadrants
            
    def get_quadrant(self):
        """
        Generates a single quadrant of the image. The quadrant is a 2D boolean
        array where some rectangles are flipped to True.
        
        Returns:
            quadrant (numpy.ndarray): A single quadrant of the image.
        """
        
        quadrant_dim = self.DIM // 2
        quadrant = np.zeros((quadrant_dim, quadrant_dim), dtype=bool)

        for limit in self.limits:
            x_min, x_max, y_min, y_max = limit
            quadrant[x_min:x_max, y_min:y_max] = True

        return np.fliplr(quadrant)
            
    
    def get_random_quadrant(self):
        """
        Generates a single quadrant of the image. The quadrant is a 2D boolean
        array where some rectangles are flipped to True.
        
        Returns:
            quadrant (numpy.ndarray): A single quadrant of the image.
        """
        
        quadrant_dim = self.DIM // 2
        quadrant = np.zeros((quadrant_dim, quadrant_dim), dtype=bool)
        
        # Define the allowed area
        allowed_dim = quadrant_dim - self.MARGIN
        max_pos = allowed_dim - self.MINSIZE
        
        self.limits= np.zeros((self.NUM_DROPS, 4)).astype(int)

        for i in range(self.NUM_DROPS):
            x_min = np.random.randint(0, max_pos + 1) + self.MARGIN
            y_min = np.random.randint(0, max_pos + 1)

            x_max = x_min + self.MINSIZE + np.random.randint(0, max_pos - x_min + self.MARGIN + 1)
            y_max = y_min + self.MINSIZE + np.random.randint(0, max_pos - y_min + 1)
            
            self.limits[i] = x_min, x_max, y_min, y_max

            quadrant[x_min:x_max, y_min:y_max] = True

        return np.fliplr(quadrant)


class Cross(Shape2D):
    """
    Generates a cross/plus-sign shaped metasurface topology.
    
    Creates a symmetric cross pattern with specified arm dimensions.
    This is a geometric primitive useful for basic metasurface designs
    and parameter sweeping studies.
    """
    
    def __init__(self, lx=0.3, ly=0.8, dim=1001):
        """
        Initialize Cross shape with specified arm dimensions.
        
        Parameters:
        - lx: float [0,1], horizontal arm width as fraction of total dimension
        - ly: float [0,1], vertical arm height as fraction of total dimension  
        - dim: int, output image dimension (creates dim x dim array)
        
        The cross is centered in the image with arms extending symmetrically.
        """
        super().__init__(dim)  # Initialize parent Shape2D class
        self.dim = dim
        self.lx = lx  # Horizontal arm width fraction
        self.ly = ly  # Vertical arm height fraction
        self.constructCross()

    def constructCross(self):
        """
        Generate the cross pattern in the binary image array.
        
        Creates a symmetric cross by setting pixels to True in two overlapping
        rectangular regions: one horizontal and one vertical, both centered
        in the image domain.
        """
        # Convert fractional dimensions to pixel counts
        lx_pixels = int(self.dim * self.lx)  # Horizontal arm width in pixels
        ly_pixels = int(self.dim * self.ly)  # Vertical arm height in pixels

        # Calculate starting and ending indices for centered positioning
        lx_start = (self.dim - lx_pixels) // 2
        lx_end = lx_start + lx_pixels
        ly_start = (self.dim - ly_pixels) // 2
        ly_end = ly_start + ly_pixels

        # Create cross by setting overlapping rectangular regions to True
        self.img[ly_start:ly_end, lx_start:lx_end] = 1  # Vertical arm
        self.img[lx_start:lx_end, ly_start:ly_end] = 1  # Horizontal arm


class Ellipse(Shape2D):
    """
    Generates an elliptical metasurface topology.
    
    Creates a filled ellipse with specified semi-major and semi-minor axes.
    Useful for studying circular and elliptical resonators in metasurface
    applications.
    """
    
    def __init__(self, a=0.4, b=0.2, dim=1001):
        """
        Initialize Ellipse shape with specified axis lengths.
        
        Parameters:
        - a: float [0,1], semi-major axis length as fraction of image dimension
        - b: float [0,1], semi-minor axis length as fraction of image dimension
        - dim: int, output image dimension (creates dim x dim array)
        
        The ellipse is centered in the image domain.
        """
        super().__init__(dim)
        self.dim = dim
        self.a = a  # Semi-major axis fraction
        self.b = b  # Semi-minor axis fraction
        self.constructEllipse()

    def constructEllipse(self):
        """
        Generate the elliptical pattern using the standard ellipse equation.
        
        Uses the mathematical definition: (x-cx)²/a² + (y-cy)²/b² ≤ 1
        where (cx,cy) is the center and a,b are the semi-axes lengths.
        All pixels satisfying this condition are set to True.
        """
        # Define ellipse center at image center
        x_center = self.dim // 2
        y_center = self.dim // 2
        
        # Convert fractional axes to pixel dimensions
        a_pixels = int(self.dim * self.a)
        b_pixels = int(self.dim * self.b)

        # Create coordinate grids for vectorized computation
        y, x = np.ogrid[:self.dim, :self.dim]
        
        # Apply ellipse equation to create boolean mask
        mask = ((x - x_center) ** 2) / (a_pixels ** 2) + ((y - y_center) ** 2) / (b_pixels ** 2) <= 1
        
        # Set ellipse pixels to True
        self.img[mask] = 1
        
    
class Rectangle(Shape2D):
    """
    Generates a rectangular metasurface topology.
    
    Creates a filled rectangle with specified width and height dimensions.
    This is a fundamental geometric primitive for metasurface design and
    serves as a building block for more complex structures.
    """
    
    def __init__(self, lx=0.8, ly=0.3, dim=1001):
        """
        Initialize Rectangle shape with specified dimensions.
        
        Parameters:
        - lx: float [0,1], rectangle width as fraction of total image dimension
        - ly: float [0,1], rectangle height as fraction of total image dimension
        - dim: int, output image dimension (creates dim x dim array)
        
        The rectangle is centered in the image domain.
        """
        super().__init__(dim)
        self.dim = dim
        self.lx = lx  # Width fraction
        self.ly = ly  # Height fraction
        self.constructRectangle()

    def constructRectangle(self):
        """
        Generate the rectangular pattern in the binary image array.
        
        Creates a centered rectangle by defining start and end indices
        for both dimensions and setting the enclosed region to True.
        """
        # Convert fractional dimensions to pixel counts
        lx_pixels = int(self.dim * self.lx)  # Width in pixels
        ly_pixels = int(self.dim * self.ly)  # Height in pixels

        # Calculate starting and ending indices for centered positioning
        lx_start = (self.dim - lx_pixels) // 2
        lx_end = lx_start + lx_pixels
        ly_start = (self.dim - ly_pixels) // 2
        ly_end = ly_start + ly_pixels

        # Set rectangular region to True
        self.img[ly_start:ly_end, lx_start:lx_end] = 1
        

class LShape(Shape2D):
    """
    Generates an L-shaped metasurface topology.
    
    Creates an L-shaped structure consisting of two perpendicular rectangular
    arms joined at one corner. Useful for studying asymmetric resonators and
    chiral metasurface elements.
    """
    
    def __init__(self, arm_length=0.4, arm_width=0.15, dim=1001):
        """
        Initialize L-shape with specified arm dimensions.
        
        Parameters:
        - arm_length: float [0,1], length of each arm as fraction of image dimension
        - arm_width: float [0,1], width of each arm as fraction of image dimension
        - dim: int, output image dimension (creates dim x dim array)
        
        The L-shape is positioned with its corner at the image center.
        """
        super().__init__(dim)
        self.dim = dim
        self.arm_length = arm_length  # Length of each arm as fraction
        self.arm_width = arm_width    # Width of each arm as fraction
        self.constructLShape()

    def constructLShape(self):
        """
        Generate the L-shaped pattern by creating two overlapping rectangles.
        
        Constructs the L-shape using two rectangular regions:
        1. Horizontal arm extending rightward from center
        2. Vertical arm extending downward from center
        The arms overlap at their junction to form the L-shape.
        """
        # Convert fractional dimensions to pixel counts
        length_pixels = int(self.dim * self.arm_length)
        width_pixels = int(self.dim * self.arm_width)

        # Calculate starting position (bottom-left corner of L)
        start_x = self.dim // 2 - length_pixels // 2
        start_y = self.dim // 2 - length_pixels // 2

        # Create the L-shape using two rectangular regions
        # Horizontal arm (bottom part of L)
        self.img[start_y:start_y+width_pixels, start_x:start_x+length_pixels] = 1
        # Vertical arm (left part of L)  
        self.img[start_y:start_y+length_pixels, start_x:start_x+width_pixels] = 1


class SplitRing(Shape2D):
    """
    Generates a split-ring resonator (SRR) metasurface topology.
    
    Creates a ring structure with a gap, which is fundamental in metamaterial
    design. Split-ring resonators exhibit strong magnetic resonances and are
    widely used in negative-index metamaterials and metasurfaces.
    """
    
    def __init__(self, outer_radius=0.4, inner_radius=0.25, gap_angle=60, dim=1001):
        """
        Initialize split-ring with specified geometric parameters.
        
        Parameters:
        - outer_radius: float [0,1], outer ring radius as fraction of image dimension
        - inner_radius: float [0,1], inner ring radius as fraction of image dimension  
        - gap_angle: float, angular width of the gap in degrees
        - dim: int, output image dimension (creates dim x dim array)
        
        The split-ring is centered in the image with the gap oriented along
        the positive x-axis (0 degrees).
        """
        super().__init__(dim)
        self.dim = dim
        self.outer_radius = outer_radius  # Outer radius fraction
        self.inner_radius = inner_radius  # Inner radius fraction
        self.gap_angle = gap_angle        # Gap angle in degrees
        self.constructSplitRing()

    def constructSplitRing(self):
        """
        Generate the split-ring pattern using polar coordinate geometry.
        
        Creates the split-ring by:
        1. Defining an annular (ring) region between inner and outer radii
        2. Removing a angular sector to create the gap
        3. Using vectorized operations for efficient computation
        """
        center = self.dim // 2
        
        # Convert fractional radii to pixel dimensions
        outer_radius_pixels = int(self.dim * self.outer_radius / 2)
        inner_radius_pixels = int(self.dim * self.inner_radius / 2)

        # Create coordinate grids for vectorized computation
        y, x = np.ogrid[:self.dim, :self.dim]

        # Calculate distance from center for each pixel
        dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)

        # Create ring mask (pixels between inner and outer radius)
        ring_mask = (dist_from_center >= inner_radius_pixels) & (dist_from_center <= outer_radius_pixels)

        # Calculate angle for each pixel (in degrees, -180 to +180)
        angles = np.arctan2(y - center, x - center) * 180 / np.pi

        # Create gap mask (pixels within the gap angle range)
        gap_mask = (angles >= -self.gap_angle/2) & (angles <= self.gap_angle/2)

        # Apply masks: ring region minus gap region
        self.img[ring_mask & ~gap_mask] = 1


class VShape(Shape2D):
    """
    Generates a V-shaped metasurface topology.
    
    Creates a V-shaped structure consisting of two angled arms meeting at a point.
    Useful for studying directional scattering, polarization conversion, and
    asymmetric metasurface responses.
    """
    
    def __init__(self, arm_length=0.4, arm_width=0.15, angle=60, dim=1001):
        """
        Initialize V-shape with specified geometric parameters.
        
        Parameters:
        - arm_length: float [0,1], length of each arm as fraction of image dimension
        - arm_width: float [0,1], width of each arm as fraction of image dimension
        - angle: float, total angle between the two arms in degrees
        - dim: int, output image dimension (creates dim x dim array)
        
        The V-shape is centered with arms extending upward and outward.
        """
        super().__init__(dim)
        self.dim = dim
        self.arm_length = arm_length  # Length of each arm as fraction
        self.arm_width = arm_width    # Width of each arm as fraction  
        self.angle = angle            # Angle between arms in degrees
        self.constructVShape()

    def constructVShape(self):
        """
        Generate the V-shaped pattern using line geometry and distance calculations.
        
        Creates the V-shape by:
        1. Defining two line segments representing the centerlines of each arm
        2. For each pixel, calculating distance to each line segment
        3. Setting pixels within arm_width distance of either line to True
        4. Limiting the arms to the specified arm_length
        
        Uses analytical geometry for precise line-to-point distance calculations.
        """
        center = self.dim // 2
        length_pixels = int(self.dim * self.arm_length)
        width_pixels = int(self.dim * self.arm_width // 2)  # Half-width for symmetric arms

        # Calculate angles for each arm (in radians)
        # Arms are symmetric about vertical axis
        half_angle = self.angle / 2
        angle1 = (90 - half_angle) * np.pi / 180  # Left arm angle
        angle2 = (90 + half_angle) * np.pi / 180  # Right arm angle

        # Create coordinate grids relative to center
        y, x = np.ogrid[:self.dim, :self.dim]
        x = x - center
        y = center - y  # Flip y to match standard coordinate system (y increases upward)

        # Calculate distance from each pixel to each arm's centerline
        # Using line equation: ax + by + c = 0
        # Distance = |ax + by + c| / sqrt(a² + b²)
        
        # Arm 1 (left arm)
        a1 = np.sin(angle1)   # Line coefficients for arm 1
        b1 = -np.cos(angle1)
        dist1 = np.abs(a1*x + b1*y) / np.sqrt(a1**2 + b1**2)

        # Arm 2 (right arm)  
        a2 = np.sin(angle2)   # Line coefficients for arm 2
        b2 = -np.cos(angle2)
        dist2 = np.abs(a2*x + b2*y) / np.sqrt(a2**2 + b2**2)

        # Calculate projection distance along each arm (for length limiting)
        proj1 = x*np.cos(angle1) + y*np.sin(angle1)  # Distance along arm 1
        proj2 = x*np.cos(angle2) + y*np.sin(angle2)  # Distance along arm 2

        # Create masks for each arm
        # Pixel is part of arm if: within width AND within length AND in correct direction
        arm1_mask = (dist1 < width_pixels) & (proj1 > 0) & (proj1 < length_pixels)
        arm2_mask = (dist2 < width_pixels) & (proj2 > 0) & (proj2 < length_pixels)

        # Combine both arms using logical OR
        self.img[arm1_mask | arm2_mask] = 1
        
    
class HeightmapSlice(Shape2D):
    """
    Creates 2D topologies by slicing through 3D height landscapes generated from 
    interpolated control points. Supports both random generation and deterministic 
    recreation for optimization workflows.
    """
    
    def __init__(self, height_values=None, grid_size=7, slice_height=0.5, 
                 interpolation='cubic', dim=64, smoothing=0.1):
        """
        Initialize HeightmapSlice instance.
        
        Parameters:
        - grid_size: int, creates N x N grid of random height points (random mode)
        - height_values: 1D array, predefined height values for grid points (deterministic mode)
        - slice_height: float [0,1], height at which to slice the 3D landscape
        - interpolation: str, interpolation method ('linear', 'cubic', 'quintic')
        - dim: int, output image dimension
        - smoothing: float, smoothing factor for interpolation
        """
        super().__init__(dim)
        self.grid_size = grid_size
        self.height_values = height_values
        self.slice_height = slice_height
        self.interpolation = interpolation
        self.smoothing = smoothing
        
        # Validate input modes
        if height_values is not None and grid_size is None:
            # Deterministic mode - infer grid size from height values
            self.grid_size = int(np.sqrt(len(height_values)))
            if self.grid_size ** 2 != len(height_values):
                raise ValueError("height_values length must be a perfect square")
            self.height_values = height_values.reshape(self.grid_size, self.grid_size)
            self.generate_landscape_from_heights()
            
        elif grid_size is not None and height_values is None:
            # Random mode - generate random heights
            self.generate_random_heights()
            self.generate_landscape_from_heights()
            
        else:
            raise ValueError("Either grid_size or height_values should be provided, not both")
    
    def generate_random_heights(self):
        """Generate random height values for the grid points."""
        self.height_values = np.random.rand(self.grid_size, self.grid_size)
    
    def generate_landscape_from_heights(self):
        """
        Create the 3D landscape from height values and slice it to create 2D topology.
        """
        from scipy.interpolate import RectBivariateSpline
        
        # Create coordinate grids for the control points
        grid_coords = np.linspace(0, 1, self.grid_size)
        
        # Create high-resolution coordinate grids for interpolation
        high_res_coords = np.linspace(0, 1, self.DIM)
        
        # Create the interpolation function
        if self.interpolation == 'linear':
            kx = ky = 1
        elif self.interpolation == 'cubic':
            kx = ky = 3
        elif self.interpolation == 'quintic':
            kx = ky = 5
        else:
            raise ValueError("interpolation must be 'linear', 'cubic', or 'quintic'")
        
        # Ensure we have enough points for the interpolation order
        if self.grid_size <= kx:
            kx = ky = min(self.grid_size - 1, 1)
        
        # Create the spline interpolator
        spline = RectBivariateSpline(grid_coords, grid_coords, self.height_values, 
                                   kx=kx, ky=ky, s=self.smoothing)
        
        # Evaluate the spline on the high-resolution grid
        height_surface = spline(high_res_coords, high_res_coords)
        
        # Create the binary topology by slicing at the specified height
        self.img = (height_surface >= self.slice_height).astype(bool)
    
    def get_height_surface(self):
        """
        Return the full height surface for visualization or analysis.
        Useful for understanding the 3D landscape before slicing.
        """
        from scipy.interpolate import RectBivariateSpline
        
        grid_coords = np.linspace(0, 1, self.grid_size)
        high_res_coords = np.linspace(0, 1, self.DIM)
        
        if self.interpolation == 'linear':
            kx = ky = 1
        elif self.interpolation == 'cubic':
            kx = ky = 3
        elif self.interpolation == 'quintic':
            kx = ky = 5
        
        if self.grid_size <= kx:
            kx = ky = min(self.grid_size - 1, 1)
        
        spline = RectBivariateSpline(grid_coords, grid_coords, self.height_values, 
                                   kx=kx, ky=ky, s=self.smoothing)
        
        return spline(high_res_coords, high_res_coords)
    
    def visualize_3d_landscape(self, figsize=(12, 5), savefig=True, name='heightmapslice_3d'):
        """
        Visualize both the 3D height landscape and the resulting 2D slice.
        """
        height_surface = self.get_height_surface()
        
        fig = plt.figure(figsize=figsize)
        
        # 3D landscape plot
        ax1 = fig.add_subplot(121, projection='3d')
        x = np.linspace(0, 1, self.DIM)
        y = np.linspace(0, 1, self.DIM)
        X, Y = np.meshgrid(x, y)
        
        surf = ax1.plot_surface(X, Y, height_surface, cmap='terrain', alpha=0.8)
        
        # Add slice plane
        slice_plane = np.full_like(height_surface, self.slice_height)
        ax1.plot_surface(X, Y, slice_plane, alpha=0.3, color='red')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Height')
        ax1.set_title('3D Height Landscape with Slice Plane')
        
        # 2D slice result
        ax2 = fig.add_subplot(122)
        ax2.imshow(self.img, cmap=self.cmap, origin='upper')
        ax2.set_title(f'2D Slice at height {self.slice_height}')
        ax2.axis('off')
        
        plt.tight_layout()
        
        if savefig:
            fig.savefig(name + '.png', bbox_inches='tight')
            plt.clf()
        else: plt.show()
    
    def get_shape_data(self):
        """Return all the class variables for analysis or optimization."""
        return {
            'grid_size': self.grid_size,
            'height_values': self.height_values.flatten(),
            'slice_height': self.slice_height,
            'interpolation': self.interpolation,
            'smoothing': self.smoothing,
            'img': self.img
        }
    
    
class DiffusionAggregation(Shape2D):
    """
    Creates 2D topologies using Diffusion-Limited Aggregation (DLA) algorithm.
    Particles undergo random walks and aggregate when they contact existing structures,
    forming fractal-like branching patterns similar to lightning or biological growth.
    """
    
    def __init__(self, particle_paths=None, num_particles=1_000, seed_type='cross', 
                 seed_size=3, stickiness=0.3, boundary='square', spawn_radius=0.6,
                 max_walk_steps=10_000, dim=64):
        """
        Initialize DiffusionAggregation instance.
        
        Parameters:
        - particle_paths: list of arrays, predefined particle trajectories (deterministic mode)
        - num_particles: int, number of particles to simulate (random mode)
        - seed_type: str, initial seed shape ('point', 'line', 'cross', 'circle')
        - seed_size: int, size of initial seed in pixels
        - stickiness: float [0,1], probability of sticking when contacting aggregate
        - boundary: str, particle spawn boundary ('circular', 'square', 'top', 'bottom')
        - spawn_radius: float, radius for particle spawning (as fraction of domain)
        - max_walk_steps: int, maximum steps before particle is removed
        - dim: int, output image dimension
        """
        super().__init__(dim)
        self.particle_paths = particle_paths
        self.num_particles = num_particles
        self.seed_type = seed_type
        self.seed_size = seed_size
        self.stickiness = stickiness
        self.boundary = boundary
        self.spawn_radius = spawn_radius
        self.max_walk_steps = max_walk_steps
        
        # Initialize the aggregate grid
        self.aggregate = np.zeros((self.DIM, self.DIM), dtype=bool)
        self.growth_order = []  # Track order of particle aggregation
        
        # Create initial seed
        self.create_seed()
        
        if particle_paths is not None:
            # Deterministic mode - replay exact particle paths
            self.aggregate_from_paths()
        else:
            # Random mode - simulate DLA process
            self.simulate_dla()
        
        # Set final image
        self.img = self.aggregate.copy()
    
    def create_seed(self):
        """Create the initial seed aggregate based on seed_type."""
        center = self.DIM // 2
        
        if self.seed_type == 'point':
            self.aggregate[center, center] = True
            
        elif self.seed_type == 'line':
            start = center - self.seed_size // 2
            end = start + self.seed_size
            self.aggregate[center, start:end] = True
            
        elif self.seed_type == 'cross':
            start = center - self.seed_size // 2
            end = start + self.seed_size
            self.aggregate[center, start:end] = True  # horizontal
            self.aggregate[start:end, center] = True  # vertical
            
        elif self.seed_type == 'circle':
            y, x = np.ogrid[:self.DIM, :self.DIM]
            mask = (x - center)**2 + (y - center)**2 <= self.seed_size**2
            self.aggregate[mask] = True
    
    def get_spawn_position(self):
        """Generate random spawn position based on boundary type."""
        center = self.DIM // 2
        spawn_dist = int(self.spawn_radius * self.DIM / 2)
        
        if self.boundary == 'circular':
            angle = np.random.uniform(0, 2*np.pi)
            x = center + int(spawn_dist * np.cos(angle))
            y = center + int(spawn_dist * np.sin(angle))
            
        elif self.boundary == 'square':
            side = np.random.randint(4)
            if side == 0:  # top
                x, y = np.random.randint(self.DIM), 0
            elif side == 1:  # right
                x, y = self.DIM-1, np.random.randint(self.DIM)
            elif side == 2:  # bottom
                x, y = np.random.randint(self.DIM), self.DIM-1
            else:  # left
                x, y = 0, np.random.randint(self.DIM)
                
        elif self.boundary == 'top':
            x, y = np.random.randint(self.DIM), 0
            
        elif self.boundary == 'bottom':
            x, y = np.random.randint(self.DIM), self.DIM-1
        
        # Ensure spawn position is within bounds
        x = np.clip(x, 0, self.DIM-1)
        y = np.clip(y, 0, self.DIM-1)
        
        return x, y
    
    def get_neighbors(self, x, y):
        """Get valid neighboring positions (8-connected)."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.DIM and 0 <= ny < self.DIM:
                    neighbors.append((nx, ny))
        return neighbors
    
    def is_adjacent_to_aggregate(self, x, y):
        """Check if position is adjacent to existing aggregate."""
        for nx, ny in self.get_neighbors(x, y):
            if self.aggregate[ny, nx]:
                return True
        return False
    
    def random_walk_step(self, x, y):
        """Take one random walk step."""
        neighbors = self.get_neighbors(x, y)
        if neighbors:
            return neighbors[np.random.randint(len(neighbors))]
        return x, y
    
    def simulate_particle(self):
        """Simulate one particle's random walk until aggregation or timeout."""
        x, y = self.get_spawn_position()
        path = [(x, y)]
        
        for step in range(self.max_walk_steps):
            # Check if adjacent to aggregate
            if self.is_adjacent_to_aggregate(x, y):
                # Stickiness check
                if np.random.random() < self.stickiness:
                    # Aggregate the particle
                    self.aggregate[y, x] = True
                    self.growth_order.append(len(self.growth_order))
                    return path
            
            # Take random walk step
            x, y = self.random_walk_step(x, y)
            path.append((x, y))
            
            # Check if particle wandered too far (optional optimization)
            center = self.DIM // 2
            if (x - center)**2 + (y - center)**2 > (self.spawn_radius * self.DIM)**2:
                break
        
        return None  # Particle didn't aggregate
    
    def simulate_dla(self):
        """Run the full DLA simulation."""
        aggregated_count = 0
        attempts = 0
        max_attempts = self.num_particles * 10  # Prevent infinite loops
        
        while aggregated_count < self.num_particles and attempts < max_attempts:
            path = self.simulate_particle()
            attempts += 1
            
            if path is not None:
                aggregated_count += 1
                # Store path for deterministic recreation
                if not hasattr(self, 'successful_paths'):
                    self.successful_paths = []
                self.successful_paths.append(np.array(path))
    
    def aggregate_from_paths(self):
        """Deterministic mode: recreate aggregate from predefined paths."""
        for i, path in enumerate(self.particle_paths):
            if len(path) > 0:
                # Get final position from path
                final_x, final_y = path[-1]
                if 0 <= final_x < self.DIM and 0 <= final_y < self.DIM:
                    self.aggregate[final_y, final_x] = True
                    self.growth_order.append(i)
    
    def visualize_growth_process(self, figsize=(15, 5)):
        """Visualize the DLA growth process and final result."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Show seed
        axes[0].imshow(self.aggregate * 0, cmap='gray', origin='upper')
        seed_only = np.zeros_like(self.aggregate)
        center = self.DIM // 2
        if self.seed_type == 'point':
            seed_only[center, center] = True
        axes[0].imshow(seed_only, cmap='Reds', alpha=0.8, origin='upper')
        axes[0].set_title('Initial Seed')
        axes[0].axis('off')
        
        # Show intermediate growth (if we have growth order data)
        if hasattr(self, 'successful_paths') and len(self.successful_paths) > 0:
            # Show paths of first few particles
            axes[1].imshow(np.zeros_like(self.aggregate), cmap='gray', origin='upper')
            colors = plt.cm.viridis(np.linspace(0, 1, min(10, len(self.successful_paths))))
            
            for i, path in enumerate(self.successful_paths[:10]):
                if len(path) > 1:
                    x_coords = path[:, 0]
                    y_coords = path[:, 1]
                    axes[1].plot(x_coords, y_coords, color=colors[i], alpha=0.7, linewidth=1)
                    axes[1].scatter(x_coords[-1], y_coords[-1], color=colors[i], s=20)
            
            axes[1].set_title('Particle Paths (first 10)')
            axes[1].axis('off')
        else:
            axes[1].imshow(self.aggregate, cmap='gray', origin='upper')
            axes[1].set_title('Intermediate Growth')
            axes[1].axis('off')
        
        # Show final aggregate
        axes[2].imshow(self.img, cmap='gray', origin='upper')
        axes[2].set_title(f'Final Aggregate ({np.sum(self.img)} particles)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_fractal_dimension(self):
        """Estimate fractal dimension using box-counting method."""
        # Simple box-counting implementation
        sizes = [2, 4, 8, 16, 32]
        counts = []
        
        for size in sizes:
            count = 0
            for i in range(0, self.DIM, size):
                for j in range(0, self.DIM, size):
                    box = self.img[i:i+size, j:j+size]
                    if np.any(box):
                        count += 1
            counts.append(count)
        
        # Fit log-log relationship
        log_sizes = np.log(sizes)
        log_counts = np.log(counts)
        
        # Avoid issues with zero counts
        valid_idx = np.isfinite(log_counts)
        if np.sum(valid_idx) > 1:
            slope, _ = np.polyfit(log_sizes[valid_idx], log_counts[valid_idx], 1)
            return -slope
        return None
    
    def get_shape_data(self):
        """Return all parameters for optimization/analysis."""
        data = {
            'seed_type': self.seed_type,
            'seed_size': self.seed_size,
            'num_particles': self.num_particles,
            'stickiness': self.stickiness,
            'boundary': self.boundary,
            'spawn_radius': self.spawn_radius,
            'img': self.img,
            'growth_order': self.growth_order
        }
        
        if hasattr(self, 'successful_paths'):
            data['particle_paths'] = self.successful_paths
            
        fractal_dim = self.get_fractal_dimension()
        if fractal_dim is not None:
            data['fractal_dimension'] = fractal_dim
            
        return data


class WaveInterference(Shape2D):
    """
    Creates 2D topologies using wave interference patterns from multiple point sources.
    """
    
    def __init__(self, source_positions=None, num_sources=4, frequencies=None, 
                 phases=None, amplitudes=None, frequency_range=[2.0, 8.0], 
                 phase_range=[0, 2*np.pi], amplitude_range=[0.8, 1.2], 
                 threshold=0.5, wave_type='circular', dim=64):
        """
        Initialize WaveInterference instance with improved parameters.
        
        Key changes:
        - Higher frequency range for visible patterns
        - Simplified distance decay
        - Better threshold handling
        """
        super().__init__(dim)
        self.source_positions = source_positions
        self.num_sources = num_sources
        self.frequencies = frequencies
        self.phases = phases
        self.amplitudes = amplitudes
        self.frequency_range = frequency_range
        self.phase_range = phase_range
        self.amplitude_range = amplitude_range
        self.threshold = threshold
        self.wave_type = wave_type
        
        # Generate parameters
        if source_positions is not None:
            self.num_sources = len(source_positions)
            if frequencies is None:
                self.frequencies = np.random.uniform(*frequency_range, self.num_sources)
            if phases is None:
                self.phases = np.random.uniform(*phase_range, self.num_sources)
            if amplitudes is None:
                self.amplitudes = np.random.uniform(*amplitude_range, self.num_sources)
        else:
            self.generate_random_parameters()
        
        # Generate the interference pattern
        self.generate_interference_pattern()
    
    def generate_random_parameters(self):
        """Generate random wave source parameters with better defaults."""
        # Place sources away from edges to avoid boundary effects
        margin = 0.1
        self.source_positions = np.random.uniform(margin, 1-margin, (self.num_sources, 2))
        
        # Use higher frequencies for visible patterns
        self.frequencies = np.random.uniform(*self.frequency_range, self.num_sources)
        self.phases = np.random.uniform(*self.phase_range, self.num_sources)
        self.amplitudes = np.random.uniform(*self.amplitude_range, self.num_sources)
    
    def calculate_wave_field(self, x_grid, y_grid):
        """
        Calculate interference pattern with simplified, more robust approach.
        """
        # Initialize the wave field
        wave_field = np.zeros_like(x_grid, dtype=float)
        
        # Convert grid coordinates to unit square
        x_norm = x_grid / self.DIM
        y_norm = y_grid / self.DIM
        
        # Add contribution from each source
        for i in range(self.num_sources):
            src_x, src_y = self.source_positions[i]
            
            # Calculate distance from source
            dx = x_norm - src_x
            dy = y_norm - src_y
            distance = np.sqrt(dx**2 + dy**2)
            
            # Avoid division by zero
            distance = np.maximum(distance, 1e-6)
            
            if self.wave_type == 'circular':
                # Circular wave with simple 1/sqrt(r) decay
                amplitude = self.amplitudes[i] / np.sqrt(distance + 0.01)
                # Phase includes distance-dependent term
                phase = self.phases[i] + 2 * np.pi * self.frequencies[i] * distance
                
            elif self.wave_type == 'plane':
                # Plane wave approximation
                amplitude = self.amplitudes[i] * np.ones_like(distance)
                # Phase varies linearly with position
                phase = self.phases[i] + 2 * np.pi * self.frequencies[i] * (dx * np.cos(i) + dy * np.sin(i))
            
            # Add wave contribution
            wave_field += amplitude * np.cos(phase)
        
        return wave_field
    
    def generate_interference_pattern(self):
        """Generate interference pattern with adaptive thresholding."""
        # Create coordinate grids
        x = np.arange(self.DIM)
        y = np.arange(self.DIM)
        x_grid, y_grid = np.meshgrid(x, y)
        
        # Calculate wave field
        self.interference_field = self.calculate_wave_field(x_grid, y_grid)
        
        # Adaptive thresholding based on field statistics
        field_mean = np.mean(self.interference_field)
        field_std = np.std(self.interference_field)
        
        # Use threshold relative to field statistics
        if self.threshold <= 1.0:
            # Interpret as percentile
            threshold_value = np.percentile(self.interference_field, self.threshold * 100)
        else:
            # Use absolute threshold
            threshold_value = self.threshold
        
        # Create binary pattern
        self.img = (self.interference_field >= threshold_value).astype(bool)
        
        # Store additional info for analysis
        self.field_stats = {
            'mean': field_mean,
            'std': field_std,
            'min': np.min(self.interference_field),
            'max': np.max(self.interference_field),
            'threshold_used': threshold_value,
            'fill_fraction': np.mean(self.img)
        }
    
    def visualize_interference_debug(self, figsize=(20, 4), savefig=True, name='waveinterference_debug'):
        """Debug visualization to understand what's happening."""
        fig, axes = plt.subplots(1, 5, figsize=figsize)
        
        # Source positions
        axes[0].imshow(np.zeros((self.DIM, self.DIM)), cmap='gray', origin='upper')
        src_x = self.source_positions[:, 0] * self.DIM
        src_y = self.source_positions[:, 1] * self.DIM
        colors = plt.cm.viridis(np.linspace(0, 1, self.num_sources))
        
        for i in range(self.num_sources):
            axes[0].scatter(src_x[i], src_y[i], c=[colors[i]], s=100, marker='o')
            axes[0].text(src_x[i]+2, src_y[i], f'f={self.frequencies[i]:.1f}', 
                        fontsize=8, color='white')
        axes[0].set_title('Sources')
        axes[0].axis('off')
        
        # Raw interference field
        im1 = axes[1].imshow(self.interference_field, cmap='RdBu', origin='upper')
        axes[1].set_title('Raw Field')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        # Field histogram
        axes[2].hist(self.interference_field.flatten(), bins=50, alpha=0.7)
        axes[2].axvline(self.field_stats['threshold_used'], color='red', linestyle='--', 
                       label=f"Threshold: {self.field_stats['threshold_used']:.3f}")
        axes[2].set_title('Field Distribution')
        axes[2].legend()
        
        # Thresholded field (grayscale)
        threshold_field = (self.interference_field >= self.field_stats['threshold_used']).astype(float)
        axes[3].imshow(threshold_field, cmap=self.cmap, origin='upper')
        axes[3].set_title(f"Thresholded (fill: {self.field_stats['fill_fraction']:.2f})")
        axes[3].axis('off')
        
        # Final binary result
        axes[4].imshow(self.img, cmap=self.cmap, origin='upper')
        axes[4].set_title('Final Binary')
        axes[4].axis('off')
        
        # Print stats
        print("Field Statistics:")
        for key, value in self.field_stats.items():
            print(f"  {key}: {value:.4f}")
        
        plt.tight_layout()
        
        if savefig:
            fig.savefig(name + '.png', bbox_inches='tight')
            plt.clf()
        else: plt.show()
        
        
class CellularAutomata(Shape2D):
    """
    Creates 2D topologies using cellular automata evolution from initial seed patterns.
    Supports both random generation and deterministic recreation for optimization workflows.
    Based on Conway's Game of Life and other CA rules.
    """
    
    def __init__(self, initial_state=None, seed_pattern='random', density=0.5, 
                 rule='B36/S23', generations=50, boundary='periodic', 
                 rule_params=None, dim=64):
        """
        Initialize CellularAutomata instance.
        
        Parameters:
        - initial_state: 2D array, predefined initial pattern (deterministic mode)
        - seed_pattern: str, type of initial pattern ('random', 'single', 'cross', 'glider')
        - density: float [0,1], density of initial random pattern
        - rule: str, CA rule in Born/Survive notation (e.g., 'B3/S23' for Conway's Game of Life)
        - generations: int, number of evolution steps
        - boundary: str, boundary conditions ('periodic', 'fixed', 'absorbing')
        - rule_params: dict, custom rule parameters for advanced CA types
        - dim: int, output image dimension
        """
        super().__init__(dim)
        self.initial_state = initial_state
        self.seed_pattern = seed_pattern
        self.density = density
        self.rule = rule
        self.generations = generations
        self.boundary = boundary
        self.rule_params = rule_params or {}
        
        # Parse the rule string
        self.parse_rule()
        
        # Evolution history for analysis
        self.evolution_history = []
        
        # Generate initial state
        if initial_state is not None:
            # Deterministic mode - use provided initial state
            if initial_state.shape != (self.DIM, self.DIM):
                # Resize if needed
                from scipy.ndimage import zoom
                scale_factor = self.DIM / initial_state.shape[0]
                self.current_state = zoom(initial_state.astype(float), scale_factor, order=0) > 0.5
            else:
                self.current_state = initial_state.astype(bool)
        else:
            # Random mode - generate initial pattern
            self.generate_initial_state()
        
        # Store initial state
        self.evolution_history.append(self.current_state.copy())
        
        # Evolve the cellular automaton
        self.evolve()
        
        # Set final image
        self.img = self.current_state.copy()
    
    def parse_rule(self):
        """Parse Born/Survive rule notation (e.g., 'B3/S23')."""
        if '/' in self.rule:
            parts = self.rule.split('/')
            born_part = parts[0].upper()
            survive_part = parts[1].upper()
            
            # Extract numbers for birth conditions
            if born_part.startswith('B'):
                self.birth_conditions = [int(x) for x in born_part[1:] if x.isdigit()]
            else:
                self.birth_conditions = [int(x) for x in born_part if x.isdigit()]
            
            # Extract numbers for survival conditions
            if survive_part.startswith('S'):
                self.survival_conditions = [int(x) for x in survive_part[1:] if x.isdigit()]
            else:
                self.survival_conditions = [int(x) for x in survive_part if x.isdigit()]
        else:
            # Default to Conway's Game of Life
            self.birth_conditions = [3]
            self.survival_conditions = [2, 3]
    
    def generate_initial_state(self):
        """Generate initial state based on seed_pattern."""
        self.current_state = np.zeros((self.DIM, self.DIM), dtype=bool)
        center = self.DIM // 2
        
        if self.seed_pattern == 'random':
            # Random distribution with specified density
            self.current_state = np.random.random((self.DIM, self.DIM)) < self.density
            
        elif self.seed_pattern == 'single':
            # Single cell in center
            self.current_state[center, center] = True
            
        elif self.seed_pattern == 'cross':
            # Cross pattern in center
            size = max(3, self.DIM // 10)
            start = center - size // 2
            end = start + size
            self.current_state[center, start:end] = True  # horizontal
            self.current_state[start:end, center] = True  # vertical
            
        elif self.seed_pattern == 'glider':
            # Conway's Game of Life glider pattern
            if self.DIM >= 10:
                glider = np.array([
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 1]
                ], dtype=bool)
                start_x = center - 10
                start_y = center - 10
                self.current_state[start_y:start_y+3, start_x:start_x+3] = glider
            
        elif self.seed_pattern == 'acorn':
            # Acorn pattern - creates complex evolution
            if self.DIM >= 20:
                acorn = np.array([
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [1, 1, 0, 0, 1, 1, 1]
                ], dtype=bool)
                start_x = center - 10
                start_y = center - 5
                self.current_state[start_y:start_y+3, start_x:start_x+7] = acorn
                
        elif self.seed_pattern == 'r_pentomino':
            # R-pentomino - another interesting pattern
            if self.DIM >= 10:
                r_pent = np.array([
                    [0, 1, 1],
                    [1, 1, 0],
                    [0, 1, 0]
                ], dtype=bool)
                start_x = center - 5
                start_y = center - 5
                self.current_state[start_y:start_y+3, start_x:start_x+3] = r_pent
    
    def count_neighbors(self, state):
        """Count neighbors for each cell using convolution."""
        # Define neighbor counting kernel (Moore neighborhood)
        kernel = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ])
        
        if self.boundary == 'periodic':
            # Periodic boundary conditions
            padded = np.pad(state, 1, mode='wrap')
        elif self.boundary == 'fixed':
            # Fixed boundary (edges are always False)
            padded = np.pad(state, 1, mode='constant', constant_values=False)
        else:  # absorbing
            # Absorbing boundary (same as fixed for this case)
            padded = np.pad(state, 1, mode='constant', constant_values=False)
        
        # Count neighbors using convolution
        from scipy.ndimage import convolve
        neighbor_count = convolve(padded.astype(int), kernel, mode='constant')
        
        # Remove padding
        return neighbor_count[1:-1, 1:-1]
    
    def apply_rule(self, state, neighbor_counts):
        """Apply cellular automaton rule to get next state."""
        new_state = np.zeros_like(state, dtype=bool)
        
        # Birth rule: dead cells with right number of neighbors become alive
        birth_mask = (~state) & np.isin(neighbor_counts, self.birth_conditions)
        new_state[birth_mask] = True
        
        # Survival rule: alive cells with right number of neighbors stay alive
        survival_mask = state & np.isin(neighbor_counts, self.survival_conditions)
        new_state[survival_mask] = True
        
        return new_state
    
    def evolve(self):
        """Evolve the cellular automaton for specified number of generations."""
        for generation in range(self.generations):
            # Count neighbors
            neighbor_counts = self.count_neighbors(self.current_state)
            
            # Apply rule to get next state
            self.current_state = self.apply_rule(self.current_state, neighbor_counts)
            
            # Store in history
            self.evolution_history.append(self.current_state.copy())
            
            # Check for extinction or stability (optional optimization)
            if generation > 0:
                if np.array_equal(self.current_state, self.evolution_history[-2]):
                    # Stable state reached
                    break
                elif not np.any(self.current_state):
                    # All cells died
                    break
    
    def visualize_evolution(self, figsize=(16, 4), show_generations=None):
        """Visualize the evolution process."""
        if show_generations is None:
            # Show key generations: initial, quarter, half, three-quarter, final
            total_gens = len(self.evolution_history)
            indices = [0, total_gens//4, total_gens//2, 3*total_gens//4, total_gens-1]
            indices = [i for i in indices if i < total_gens]
            show_generations = list(set(indices))  # Remove duplicates
        
        n_plots = len(show_generations)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        
        if n_plots == 1:
            axes = [axes]
        
        for i, gen_idx in enumerate(show_generations):
            if gen_idx < len(self.evolution_history):
                axes[i].imshow(self.evolution_history[gen_idx], cmap=self.cmap, origin='upper')
                axes[i].set_title(f'Generation {gen_idx}')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_evolution_properties(self):
        """Analyze properties of the evolution."""
        properties = {}
        
        # Population over time
        populations = [np.sum(state) for state in self.evolution_history]
        properties['population_history'] = populations
        properties['final_population'] = populations[-1]
        properties['max_population'] = max(populations)
        properties['population_stability'] = len(populations) - 1  # Generations until stability
        
        # Fill fraction
        properties['final_fill_fraction'] = populations[-1] / (self.DIM * self.DIM)
        
        # Pattern complexity (using entropy)
        final_state = self.evolution_history[-1]
        if np.any(final_state):
            # Calculate spatial entropy
            from scipy.stats import entropy
            # Simple measure: entropy of row sums and column sums
            row_sums = np.sum(final_state, axis=1)
            col_sums = np.sum(final_state, axis=0)
            row_entropy = entropy(row_sums + 1e-10)  # Add small value to avoid log(0)
            col_entropy = entropy(col_sums + 1e-10)
            properties['spatial_complexity'] = (row_entropy + col_entropy) / 2
        else:
            properties['spatial_complexity'] = 0
        
        # Stability detection
        if len(self.evolution_history) >= 2:
            properties['is_stable'] = np.array_equal(
                self.evolution_history[-1], 
                self.evolution_history[-2]
            )
        else:
            properties['is_stable'] = True
        
        return properties
    
    def get_shape_data(self):
        """Return all parameters for optimization/analysis."""
        properties = self.analyze_evolution_properties()
        
        return {
            'initial_state': self.evolution_history[0] if self.evolution_history else None,
            'seed_pattern': self.seed_pattern,
            'density': self.density,
            'rule': self.rule,
            'birth_conditions': self.birth_conditions,
            'survival_conditions': self.survival_conditions,
            'generations': self.generations,
            'boundary': self.boundary,
            'evolution_history': self.evolution_history,
            'final_state': self.img,
            'properties': properties
        }