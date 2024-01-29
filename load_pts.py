import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN

from generate_cuboid import normalize_point_cloud, points_to_pcd

def load_pts(point_cloud_file):
    """
    Load a point cloud as a (N, 3) np.ndarray
    """
    # Load the .pts file
    point_cloud = o3d.io.read_point_cloud(point_cloud_file)

    # Convert to NumPy array
    return np.asarray(point_cloud.points)

def find_cube_clouds(whole_cloud, radius_estimate=8.0, min_points=150):
    """
    Given a point cloud as a (N, 3) np.ndarray, find 2 alignment
    cubes within that point cloud and return them as 2 sub-clouds
    in a tuple.
    """
    # Assuming 'points' is your Nx3 NumPy array from the point cloud
    # Set eps to your estimated cluster radius and experiment with min_samples
    clustering = DBSCAN(eps=radius_estimate, min_samples=min_points).fit(whole_cloud)

    # The cluster labels for each point
    labels = clustering.labels_

    # Number of clusters, excluding noise if present
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    clusters = [whole_cloud[labels == n] for n in range(n_clusters_) if whole_cloud[labels == n].shape[0] < 1000 ]

    return clusters

def get_cubes_from_file(point_cloud_file):
    return find_cube_clouds(load_pts(point_cloud_file))

def viz_cubes(point_cloud_file):
    cubes = get_cubes_from_file(point_cloud_file)

    print(f"cubes found {len(cubes)}")

    for cube in cubes:
        print(f"cube shape {cube.shape}")
        normpoints = normalize_point_cloud(cube)[0]

        # Visualize the point cloud
        o3d.visualization.draw_geometries([points_to_pcd(normpoints)])

if __name__ == "__main__":
    viz_cubes("3-transformedModels_3.pts")