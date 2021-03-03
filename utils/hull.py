import math


def distance(p1, p2):
    """
    Calculates the distance between two points.

    :param p1, p2: points
    :return: distance between points
    """
    p1x, p1y = p1
    p2x, p2y = p2

    return math.sqrt((p1x - p2x) ** 2 + (p2y - p2y) ** 2)


def knn(points, p, k):
    """
    Calculates the k nearest neighbours of a point.

    :param points: list of points
    :param p: reference point
    :param k: amount of neighbours
    :return: list of k neighbours
    """
    return sorted(points, key=lambda x: distance(p, x))[:k]


def intersects(p1, p2, p3, p4):
    """
    Checks if the lines [p1, p2] and [p3, p4] intersect.

    :param p1, p2: line
    :param p3, p4: line
    :return: lines intersect
    """
    p0x, p0y = p1
    p1x, p1y = p2
    p2x, p2y = p3
    p3x, p3y = p4

    s10x = p1x - p0x
    s10y = p1y - p0y
    s32x = p3x - p2x
    s32y = p3y - p2y

    denom = s10x * s32y - s32x * s10y
    if denom == 0:
        return False

    denom_positive = denom > 0
    s02x = p0x - p2x
    s02y = p0y - p2y
    s_numer = s10x * s02y - s10y * s02x
    if (s_numer < 0) == denom_positive:
        return False

    t_numer = s32x * s02y - s32y * s02x
    if (t_numer < 0) == denom_positive:
        return False

    if (s_numer > denom) == denom_positive or (t_numer > denom) == denom_positive:
        return False

    t = t_numer / denom
    x = p0x + (t * s10x)
    y = p0y + (t * s10y)

    return (x, y) not in [p1, p2, p3, p4]


def angle(p1, p2, previous=0):
    """
    Calculates the angle between two points.

    :param p1, p2: points
    :param previous: previous angle
    :return: angle
    """
    p1x, p1y = p1
    p2x, p2y = p2

    return (math.atan2(p1y - p2y, p1x - p2x) - previous) % (math.pi * 2) - math.pi


def point_in_polygon(point, polygon):
    """
    Checks if a point is inside a polygon.

    :param point: point
    :param polygon: polygon
    :return: point is inside polygon
    """
    px, py = point

    size = len(polygon)
    for i in range(size):
        p1x, p1y = polygon[i]
        p2x, p2y = polygon[(i + 1) % size]
        if min(p1x, p2x) < px <= max(p1x, p2x):
            p = p1y - p2y
            q = p1x - p2x
            y = (px - p1x) * p / q + p1y
            if y < py:
                return True

    return False


def concave(points, k=3):
    """
    Calculates the concave hull for a list of points. Each point is a tuple
    containing the x- and y-coordinate. k defines the number of considered
    neighbours.

    :param points: list of points
    :param k: considered neighbours
    :return: concave hull
    """
    dataset = list(set(points))  # Remove duplicates
    if len(dataset) < 3:
        raise Exception("Dataset length cannot be smaller than 3")
    if len(dataset) == 3:
        return dataset  # Points are a polygon already
    
    k = min(max(k, 3), len(dataset) - 1)  # Make sure that k neighbours can be found

    first = current = min(dataset, key=lambda x: x[1])
    hull = [first]  # Initialize hull
    dataset.remove(first)  # Remove processed point
    previous_angle = 0

    while (current != first or len(hull) == 1) and len(dataset) > 0:
        if len(hull) == 3:
            dataset.append(first)  # Add first point again

        neighbours = knn(dataset, current, k)
        c_points = sorted(neighbours, key=lambda x: -angle(x, current, previous_angle))

        its = True
        i = -1
        while its and i < len(c_points) - 1:
            i += 1
            last_point = 1 if c_points[i] == first else 0
            j = 1
            its = False
            while not its and j < len(hull) - last_point:
                its = intersects(hull[-1], c_points[i], hull[-j - 1], hull[-j])
                j += 1

        if its:  # All points intersect, try again with higher a number of neighbours
            return concave(points, k + 1)

        previous_angle = angle(c_points[i], current)
        current = c_points[i]
        hull.append(current)  # Valid candidate was found
        dataset.remove(current)

    for point in dataset:
        if not point_in_polygon(point, hull):
            return concave(points, k + 1)

    return hull


def cross(o, a, b):
    """
    Calculates cross between two vectors.

    :param o, a: vector
    :param o, b: vector
    :return: cross product
    """
    ox, oy = o
    ax, ay = a
    bx, by = b

    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def convex(points):
    """
    Calculates the concave hull for a list of points. Each point is a tuple
    containing the x- and y-coordinate.

    :param points: list of points
    :return: convex hull
    """
    dataset = sorted(set(points))  # Remove duplicates
    if len(dataset) <= 1:
        return dataset

    # Build lower hull
    lower = []
    for p in dataset:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(dataset):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]
