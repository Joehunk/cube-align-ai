import open3d as o3d
import numpy as np

def generate_cuboid_point_cloud(include_faces, randomness, num_outliers, outlier_radius):
    """
    Generates a point cloud for a cuboid with filtered outliers.

    :param include_faces: A dict indicating which faces to include (top, bottom, left, right, front, back).
    :param randomness: Maximum random offset in inches to apply to each point.
    :param num_outliers: Number of outlier points to add.
    :param outlier_radius: Radius in inches around the centroid for placing outlier points.
    :return: Open3D point cloud object.
    """
    # Cuboid dimensions in inches
    width, depth, height = 3, 3, 1.5  # X, Y, Z

    # Generate points for each face
    points = []
    if include_faces.get("top"):
        points.extend([np.random.rand(200, 3) * [width, depth, 0] + [0, 0, height]])
    if include_faces.get("bottom"):
        points.extend([np.random.rand(200, 3) * [width, depth, 0]])
    if include_faces.get("left"):
        points.extend([np.random.rand(100, 3) * [0, depth, height] + [0, 0, 0]])
    if include_faces.get("right"):
        points.extend([np.random.rand(100, 3) * [0, depth, height] + [width, 0, 0]])
    if include_faces.get("front"):
        points.extend([np.random.rand(100, 3) * [width, 0, height] + [0, 0, 0]])
    if include_faces.get("back"):
        points.extend([np.random.rand(100, 3) * [width, 0, height] + [0, depth, 0]])

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

    all_points = np.vstack([all_points, filtered_outliers])

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)

    return pcd

if __name__ == "__main__":
    # Example usage
    pcd = generate_cuboid_point_cloud(
        include_faces={"top": True, "bottom": False, "left": True, "right": False, "front": True, "back": False},
        randomness=0.05,  # inches
        num_outliers=30,
        outlier_radius=4  # inches
    )

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])  # This line is for visualization and can be commented out if not needed

