import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import random

def generate_cuboid_points(face_points, randomness, num_outliers, outlier_radius):
    """
    Generates a point cloud for a cuboid with filtered outliers.

    :param face_points: A dict indicating which faces to include (top, bottom, left, right, front, back)
    with values equal to the number of points to sample on that face.
    :param randomness: Maximum random offset in inches to apply to each point.
    :param num_outliers: Number of outlier points to add.
    :param outlier_radius: Radius in inches around the centroid for placing outlier points.
    :return: Open3D point cloud object.
    """
    # Cuboid dimensions in inches
    width, depth, height = 3, 3, 1.5  # X, Y, Z

    # Generate points for each face
    points = []
    points.extend([np.random.rand(face_points.get("top", 0), 3) * [width, depth, 0] + [0, 0, height]])
    points.extend([np.random.rand(face_points.get("bottom", 0), 3) * [width, depth, 0]])
    points.extend([np.random.rand(face_points.get("left", 0), 3) * [0, depth, height] + [0, 0, 0]])
    points.extend([np.random.rand(face_points.get("right", 0), 3) * [0, depth, height] + [width, 0, 0]])
    points.extend([np.random.rand(face_points.get("front", 0), 3) * [width, 0, height] + [0, 0, 0]])
    points.extend([np.random.rand(face_points.get("back", 0), 3) * [width, 0, height] + [0, depth, 0]])

    # Combine all points
    all_points = np.vstack(points)
    all_points += np.random.uniform(-randomness, randomness, size=all_points.shape)

    # Add outliers
    centroid = [width / 2, depth / 2, height / 2]
    outliers = np.random.uniform(-1, 1, size=(num_outliers * 10, 3))  # Generate extra to account for filtering
    outliers = outliers / np.linalg.norm(outliers, axis=1)[:, np.newaxis] * np.random.uniform(0, outlier_radius, size=(num_outliers * 10, 1)) + centroid

    # Filter outliers
    filtered_outliers = [pt for pt in outliers if not (0 <= pt[0] <= width and 0 <= pt[1] <= depth and pt[2] <= height)]
    filtered_outliers = np.array(filtered_outliers[:num_outliers])  # Ensure only num_outliers are kept

    return np.vstack([all_points, filtered_outliers])

def generate_bottom_center_point():
    """
    This is the 'label' for the cuboid, i.e. what we want the network to predict
    after rotation, translation, noising, etc.

    It is always the center of the bottom face of the cube. 
    """
    return np.array([1.5, 1.5, 0])

def points_to_pcd(points):
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd

def transform_point_cloud(point_cloud, euler_angles, translation_vector):
    """
    Transform a point cloud by rotating and then translating using scipy's from_euler function.

    Parameters:
    - point_cloud (Nx3 numpy array): The point cloud to transform.
    - euler_angles (tuple): Euler angles in radians (z, y, x) for rotation.
    - translation_vector (tuple): Translation vector (x, y, z).

    Returns:
    - Nx3 numpy array: Transformed point cloud.
    """
    # Create a rotation object from Euler angles
    rotation = R.from_euler('zyx', euler_angles, degrees=False)

    # Apply rotation to the point cloud
    rotated_point_cloud = rotation.apply(point_cloud)

    # Then translate the point cloud
    transformed_point_cloud = rotated_point_cloud + translation_vector

    return transformed_point_cloud

def generate_random_cube_plus_label():
    point_density = random.randrange(50, 200)
    z_rotation = random.random() * 2.0 * math.pi
    x_rotation = random.random() * 0.1 * math.pi - 0.5 * math.pi
    y_rotation = random.random() * 0.1 * math.pi - 0.5 * math.pi

    x_translation = random.random() * 600
    y_translation = random.random() * 600
    z_translation = random.random() * 20

    # Simulate some of the side faces being in shadow
    faces = [ 'left', 'front', 'right', 'back']
    density_mult = [ 
        0.3 * random.random() + 0.2, 
        0.7 * random.random() + 0.3, 
        0.3 * random.random() + 0.2
        ]
    
    face_points = {}
    start_index = random.randrange(4)
    for i in range(random.randrange(2,4)):
        index = (i + start_index) % 4
        mult = density_mult[i]
        face_points[faces[index]] = int(point_density * mult // 2)

    face_points['top'] = int(point_density)
    points = generate_cuboid_points(
        face_points=face_points,
        randomness=0.05 + random.random() * 0.05,
        num_outliers=random.randrange(20, 40),
        outlier_radius=4 * random.random() + 8,
    )
    label = generate_bottom_center_point()
    points = np.vstack([points, label])
    points = transform_point_cloud(points, (z_rotation, y_rotation, x_rotation), (x_translation, y_translation, z_translation))
    return points

def normalize_point_cloud(points):
    # Centering
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # Scaling
    max_distance = np.max(np.sqrt(np.sum(centered_points ** 2, axis=1)))
    normalized_points = centered_points / max_distance

    return normalized_points, centroid, max_distance

def denormalize_point_cloud(normalized_points, centroid, max_distance):
    # Rescaling
    rescaled_points = normalized_points * max_distance

    # Recentering
    denormalized_points = rescaled_points + centroid

    return denormalized_points

def viz_single_cube():
    # Example usage
    points = generate_random_cube_plus_label()
    normpoints = normalize_point_cloud(points)[0]
    pcd = points_to_pcd(normpoints)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])  # This line is for visualization and can be commented out if not needed

def save_dataset(file_name, items):
    data = [ generate_random_cube_plus_label() for i in range(items) ]
    np.savez_compressed(file_name, *data)

if __name__ == "__main__":
    save_dataset('./train3_negative_xy_rotation.npz', 64000)