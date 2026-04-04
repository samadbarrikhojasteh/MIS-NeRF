# icp_function(): Apply Iterative closest point to the both 3D models(PreOperative and IntraOperative) and
# Project tumors, veins and also preop model to the intraoperative images.
# depth_compare() : Comparison surface depth between reconstructed model according to the pose and GT depth map
import cv2
from osgeo_utils.samples.build_jp2_from_xml import marker_map
from trimesh.registration import icp, procrustes
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import point_cloud_utils as pcu
import numpy as np
import pandas as pd
import os

def rotation_matrix(a, b):
    """
    Compute the rotation matrix that rotates vector a to vector b.

    :param a: The vector to rotate
    :param b: The vector to rotate to.
    :return: The rotation matrix.
    """
    import numpy as np

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (np.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = np.linalg.norm(v)
    skew_sym_mat = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return np.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s**2 + 1e-8))


def undistort_points(distorted_points, distortion_coefficients, camera_matrix):
    """
    Undistort the distorted camera parameters
    :param distorted_points:
    :param distortion_coefficients:
    :param camera_matrix:
    :return:
    """
    import numpy as np

    # Extract distortion coefficients
    k1, k2, p1, p2, k3 = distortion_coefficients

    # Extract camera matrix parameters
    f_x = camera_matrix[0, 0]
    f_y = camera_matrix[1, 1]
    c_x = camera_matrix[0, 2]
    c_y = camera_matrix[1, 2]
    # Normalize the distorted points
    normalized_points = (distorted_points - [c_x, c_y]) / [f_x, f_y]
    # Undistort the normalized points
    x = normalized_points[:, 0]
    y = normalized_points[:, 1]
    r_squared = x ** 2 + y ** 2
    radial_distortion_correction = 1 + k1 * r_squared + k2 * r_squared ** 2 + k3 * r_squared ** 3
    x_radial_correction = x * radial_distortion_correction
    y_radial_correction = y * radial_distortion_correction
    x_tangential_correction = 2 * p1 * x * y + p2 * (r_squared + 2 * x ** 2)
    y_tangential_correction = p1 * (r_squared + 2 * y ** 2) + 2 * p2 * x * y
    x_undistorted = x_radial_correction + x_tangential_correction
    y_undistorted = y_radial_correction + y_tangential_correction
    # Denormalize the undistorted points
    undistorted_points = np.column_stack(((x_undistorted * f_x) + c_x, (y_undistorted * f_y) + c_y))
    return undistorted_points

# Function to uniformly sample points from a mesh
def sample_points(mesh, num_points):
    import numpy as np
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return np.asarray(pcd.points)

def farthest_point_sampling(points, num_points):
    import numpy as np
    import scipy.spatial
    N, D = points.shape
    farthest_pts = np.zeros((num_points, D))
    farthest_pts[0] = points[np.random.randint(len(points))]
    distances = np.linalg.norm(points - farthest_pts[0], axis=1)

    for i in range(1, num_points):
        farthest_pts[i] = points[np.argmax(distances)]
        new_distances = np.linalg.norm(points - farthest_pts[i], axis=1)
        distances = np.minimum(distances, new_distances)

    return farthest_pts

def chamfer(source_path:str, target_path:str, num_points:int)-> float:
    import point_cloud_utils as pcu
    import open3d as o3d
    import numpy as np
    '''This function computs chamfer distance between uniformly selected points from source and target meshes'''
    # Load meshes using open3d
    target = o3d.io.read_triangle_mesh(target_path)
    source = o3d.io.read_triangle_mesh(source_path)
    #  # By using uniform sampling, we ensure that the point distributions are consistent across meshes
    target_uniform = sample_points(target, num_points)
    source_uniform = sample_points(source, num_points)
    # Ensure both arrays are in row-major order and float type
    target_uniform = np.ascontiguousarray(target_uniform, dtype=np.float32)
    source_uniform = np.ascontiguousarray(source_uniform, dtype=np.float32)
    # Calculate Chamfer distances using pcu
    chamfer_distance = pcu.chamfer_distance(target_uniform, source_uniform)
    return chamfer_distance


def icp_function():
    """
    Performing ICP on target and source meshes according to point clouds which have been got by clicking on the Pre_Op
    and the first image of the dataset.
    :return: Plotting 3D models.
    """

    # Load the source and target meshes
    dir = "/data1/data_in_nerfstudio/data/uterus/output_MIS"
    source_path = "/data1/data_in_nerfstudio/data/uterus/test/output_MIS/nerfacto/2024-12-09_140212/mesh/poisson_mesh2.ply"
    target_path = "/data1/data_in_nerfstudio/data/uterus/models/uterus2.obj"

    #Put path for each of them if you want register if on the RGBs
    preop_path = None
    tumor_path = None
    vein_path = None

    target_mesh, target_faces = pcu.load_mesh_vf(target_path)
    source_mesh, source_faces = pcu.load_mesh_vf(source_path)

    if tumor_path is not None: tumor_mesh = pcu.load_mesh_v(tumor_path)
    else:
        tumor_mesh = None
    if vein_path is not None: vein_mesh = pcu.load_mesh_v(vein_path)
    else:
        vein_mesh = None
    if preop_path is not None: preop_mesh = pcu.load_mesh_v(preop_path)
    else:
        preop_mesh = None


    source_before = source_mesh
    target = target_mesh
    tumor_list = []
    vein_list = []

    # Take manually landmark 3D points (Global Initialization)
    # Read camera poses (json file)
    fnames, K, distortion, posewithname = readjson(dir)
    # Convert OpenGl to OpenCV coords and c2w to w2c as well.
    w2c = nerfacto2opencv(posewithname)
    # Transform the reconstructed 3D model from world space to camera space, project it onto the image plane using camera intrinsics, and normalize to obtain pixel coordinates.
    image_coords = []
    uv = []
    for i in range(len(w2c)):
            pose = w2c[i]["frame"]
            source_hemog = np.concatenate((source_before, np.tile(np.array([[1]]), (int(len(source_before)), 1))), axis=1)
            camera_coordinates = np.dot(source_hemog, pose.T)
            image_coordinates = np.dot(camera_coordinates, K.T)
            uv_z = image_coordinates / image_coordinates[:, 2:3] #normalization
            image_coords.append(image_coordinates)
            uv.append(uv_z)

    # Put 3D landmark points of Preoperative model (4 or 5 points)
    preop = np.array(
        [[-36.926796, 5.000504, -65.496605],
         [-2.894148, -10.476115, -36.340618],
         [-2.272520, -2.539776, -74.024872],
         [33.216587, 1.612050, -53.025105]])

    #Perform landmark selection on the RGBs then produce corresponding world camera coords wrt camera each C2W
    nerfacto = landmark(dir, w2c, K, image_coords, uv)
    # Procrusting to get global initial transformation
    if nerfacto and preop is None: raise ValueError("Preop and Nerfacto 3D points are not identified")
    initial_transformation, transformed, _ = procrustes(nerfacto, preop, weights=None, reflection= False, translation= True, scale=True, return_cost=True)

    # Perform ICP registration
    # we put extra parameter(total_costs) for gathering MSE distances between a and b in each iteration.
    transformation, transformed_a, cost, total_costs = icp(source_before, target, initial=initial_transformation, threshold=1e-5, max_iterations=20)

    # convert MSE that comes from ICP(cost) into RMSE between transformed source and target
    mean_distance = np.sqrt(np.stack(total_costs))
    # we use transformed(by ICP) source to get Chamfer distance error
    source = transformed_a

    # Ensure row-major order
    source = np.ascontiguousarray(source, dtype=np.float32)
    target = np.ascontiguousarray(target, dtype=np.float32)
    # use one side Chamfer (only compute mean distance from source to target)
    chamfer_distance = one_sided_chamfer_distance(source, target)
    total_costs = np.append(mean_distance, np.array(chamfer_distance))
    costs = pd.DataFrame(total_costs)
    sequential = list(range(1, len(costs)+1))
    sequential[-1] = "chamfer Distance"
    costs.insert(0, 'ICP Iterations', sequential)
    costs.to_csv(os.path.join(dir, "ICP_errors.ods"))

    #start to register tumor and vein coordinates(if available)
    T_inv = np.linalg.inv(transformation)
    if tumor_mesh is not None:
        tumor = np.concatenate((tumor_mesh, np.ones((len(tumor_mesh), 1))), axis=1)
        tumor_transfered = np.dot(tumor, T_inv.T)[:, :3]
        tumor_list.append(tumor_transfered)
    if vein_mesh is not None :
        vein = np.concatenate((vein_mesh, np.ones((len(vein_mesh), 1))), axis=1)
        vein_transfered = np.dot(vein, T_inv.T)[:, :3]
        vein_list.append(vein_transfered)
    # register Preop model on the RGBs
    if preop_mesh is not None:
        preop_vertices = np.concatenate((preop_mesh, np.ones((len(preop_mesh), 1))), axis=1)  # Pre-Op model
        preop_transformed = np.dot(preop_vertices, T_inv.T)[:, :3]
        tumor_list.append(preop_transformed)
    if tumor_mesh is not None or vein_mesh is not None or preop_mesh is not None:
        dir_tumor = os.path.join(dir, "tumor_preops")
        os.makedirs(os.path.join(dir, "tumor_preops"), exist_ok=True)
        # save registered images
        w2img(w2c, fnames, K, tumor_list, vein_list, None, None, dir_tumor)


    # Visualize all meshes with the same size
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the target mesh vertices
    ax.scatter(target[:, 0], target[:, 1],target[:, 2], color='green', alpha=0.4, label="Pre_Ope", marker= ".")
    ax.scatter(transformed_a[:, 0], transformed_a[:, 1],transformed_a[:, 2], color='red', alpha=0.4, label="Int_Ope",marker= ".")

    # Set the same limits for all axes to ensure the same size
    min_bound_x = min(np.min(transformed_a[:, 0]), np.min(target[:, 0]))
    min_bound_y = min(np.min(transformed_a[:, 1]), np.min(target[:, 1]))
    min_bound_z = min(np.min(transformed_a[:, 2]), np.min(target[:, 2]))
    max_bound_x = max(np.max(transformed_a[:, 0]), np.max(target[:, 0]))
    max_bound_y = max(np.max(transformed_a[:, 1]), np.max(target[:, 1]))
    max_bound_z = max(np.max(transformed_a[:, 2]), np.max(target[:, 2]))
    ax.set_xlim(min_bound_x, max_bound_x)
    ax.set_ylim(min_bound_y, max_bound_y)
    ax.set_zlim(min_bound_z, max_bound_z)
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Registration Int-Ope and the Pre-Ope Model')
    ax.legend()
    plt.show()
    return ()

def one_sided_chamfer_distance(source_points, target_points):
    import point_cloud_utils as pcu
    import numpy as np
    dists, _ = pcu.k_nearest_neighbors(source_points, target_points, 1)
    return np.mean(dists)

def plot(u, v, color):
    import matplotlib.pyplot as plt
    plt.scatter(u, v, s=1, c=color, marker=".")
    return

def w2img(poses, fnames, K, dir_mesh:any, veins:None, dir_before:None, dir_after:None, dir_tumor:None):

    """ Convert world to camera and then image coords and save registered, rgb and tumor parts(if available)
    Args:
         poses: pose of each frames(images) than came from colmap
         bunch of images that can be used for mapping and saving
         K: camera parameters,
         dir_mesh: path of used mesh OR transformed tumors vertices,
         veins: path of the vein 3D mesh
         dir_before: path for the registered images before masking,
         dir_after: path for the registered images after masking
    Returns:
           Save registered, rgb and tumor components
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import trimesh
    import cv2
    import os
    if isinstance(dir_mesh, str):
        source_mesh = trimesh.load(dir_mesh)
        source = [source_mesh.vertices]
    else:
        source = dir_mesh

    for i in range(len(poses)):
        UV_tumor = []
        UV_vein = []
        for tumor in source:
            pose = poses[i]["frame"]
            tumor = np.concatenate((tumor, np.tile(np.array([[1]]), (int(len(tumor)), 1))),axis=1)
            camera_coordinates = np.dot(tumor, pose.T)
            # camera_coordinates = camera_coordinates[:, :3]
            image_coordinates = np.dot(camera_coordinates, K.T)
            # Normalize point clouds (divid by z axis_depth)
            uv_z = image_coordinates / image_coordinates[:, 2:3]
            UV_tumor.append(uv_z)
        if veins is not None:
            for vein in veins:
                pose_vein = poses[i]["frame"]
                vein = np.concatenate((vein, np.tile(np.array([[1]]), (int(len(vein)), 1))), axis=1)
                camera_coordinates = np.dot(vein, pose_vein.T)
                # camera_coordinates = camera_coordinates[:, :3]
                image_coordinates = np.dot(camera_coordinates, K.T)
                # Normalize point clouds (divid by z axis_depth)
                uv_z_vein = image_coordinates / image_coordinates[:, 2:3]
                UV_vein.append(uv_z_vein)

        # Read masks and convert to tuple pairs
        gt_dir = os.path.relpath(poses[i]["name"], 'images')
        if dir_tumor is not None:
            new_dir = os.path.dirname(dir_tumor)
        else:
            new_dir = os.path.dirname(dir_before)
        extention_png = gt_dir + ".png"
        readmask_double_png = cv2.imread(os.path.join(new_dir, "masks", extention_png))[:, :, 0]
        mask_pixels = np.where(readmask_double_png > 0.0)
        mask_pixels_stack = np.stack((mask_pixels[1], mask_pixels[0]), axis=1)
        mask_tuple = set(map(tuple, mask_pixels_stack))

        # Control whether the point clouds inside the frame or not
        h, w = readmask_double_png.shape[0], readmask_double_png.shape[1]
        total_true = uv(UV_tumor, w, h)
        total_true_vein = uv(UV_vein, w, h)
        selected_total = []
        selected_vein = []

        # Control whether the point clouds (tumors or preop model) are in the mask coordinates or not
        for t in range(0, len(total_true)):
            xx = UV_tumor[t][:, 0:1][total_true[t]]
            yy = UV_tumor[t][:, 1:2][total_true[t]]
            zz = UV_tumor[t][:, 2:3][total_true[t]]
            uv_nerf = np.stack((xx, yy, zz), axis=1).astype(int)  # (n,3)
            selected_indices = np.array(
                [idx for idx, coord in enumerate(uv_nerf[:, 0:2]) if tuple(coord) in mask_tuple])
            if len(selected_indices) != 0:
                selected_rec = uv_nerf[selected_indices]  # reconstructed depth
            else: selected_rec = uv_nerf
            #we can deactivate masking processing with put only "uv_nerf"
            selected_total.append(uv_nerf) #selected_rec
        # Control whether the point clouds (veins) are in the mask coordinates or not
        for p in range(0, len(total_true_vein)):
            xx_vein = UV_vein[p][:, 0:1][total_true_vein[p]]
            yy_vein = UV_vein[p][:, 1:2][total_true_vein[p]]
            zz_vein = UV_vein[p][:, 2:3][total_true_vein[p]]
            uv_vein = np.stack((xx_vein, yy_vein, zz_vein), axis=1).astype(int)  # (n,3)
            selected_indices_vein = np.array(
                [idx for idx, coord in enumerate(uv_vein[:, 0:2]) if tuple(coord) in mask_tuple])
            if len(selected_indices_vein) != 0:
                selected_rec_vein = uv_vein[selected_indices_vein]  # reconstructed depth
            else:
                selected_rec_vein = uv_vein
            selected_vein.append(selected_rec_vein) #selected_rec_vein

        # Plot tumors or registered object
        rgb = cv2.imread(fnames[i])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb)
        for s in range(len(total_true)): plot(selected_total[s][:, 0:1], selected_total[s][:, 1:2], "green")
        for a in range(len(total_true_vein)): plot(selected_vein[a][:, 0:1], selected_vein[a][:, 1:2], "yellow")

        if dir_tumor is not None and dir_before is None and dir_after is None:
            plt.savefig(os.path.join(dir_tumor, f'pc_image{i:04d}.png'), bbox_inches='tight')
            plt.clf()
        if dir_tumor is None and dir_after is not None and dir_before is not None:
            # plot After masking folder
            plt.savefig(os.path.join(dir_after, f'pc_image{i:04d}.png'), bbox_inches='tight')
            plt.clf()

            # Plot Before masking folder
            plt.imshow(rgb)
            for s in range(len(total_true)): plot(UV_tumor[s][:, 0:1][np.concatenate(total_true)], UV_tumor[s][:, 1:2][np.concatenate(total_true)], "green")
            plt.savefig(os.path.join(dir_before, f'pc_image{i:04d}.png'), bbox_inches='tight')
            plt.clf()

    print("Saving is Done!")
    return ()


def rmse(actual:float, pred:float):
    """
    RMSE(Root Mean Square Error)
    :param actual: GT numpy array
    :param pred: predicted numpy array
    :return: rmse value
    """
    import numpy as np
    return np.sqrt(np.square(np.subtract(actual, pred)).mean())

def landmark(dir:str, w2c:dict, K, source_coords, uv):
    """This function is for convert the RGB landmarks (from the first image) to the corresponding MIS-NeRF reconstructed model landmarks"""
    import matplotlib.pyplot as plt
    import os
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2
    import numpy as np

    dir_image = os.path.join(dir, "images")
    # select the first GT image, depth and corresponding pose
    image_folder = sorted(os.listdir(dir_image))[0]

    img = cv2.imread(os.path.join(dir_image, image_folder))
    plt.imshow(img)

    pose = w2c[0]["frame"]
    c2w = np.linalg.inv(pose)# convert w2c into c2w
    source_points = source_coords[0]
    uv_source = uv[0][:, 0:2]
    depth_z = source_points[:, 2]

    #Select some landmark points on the first RGB image

    selected_landmarks = np.array(([325, 625, 1],
                                   [980, 30, 1],
                                   [1033, 900, 1],
                                   [1650, 550, 1])).astype(float)

    z_c = []
    #select depth of the selected landmarks from source_points
    # Find the closest depth for the given (u, v)
    for i in range(0, len(selected_landmarks)):
        u = int(selected_landmarks[i][0])
        v = int(selected_landmarks[i][1])
        distances = np.linalg.norm(uv_source - np.array([u, v]), axis=1)
        closest_idx = np.argmin(distances)
        z_c.append(depth_z[closest_idx])
    z_depth = np.stack(z_c)

    camera_land = np.dot(selected_landmarks, np.linalg.inv(K[:, 0:3]).T)
    #scale by depth
    camera_land_depth = camera_land * z_depth[:, None]
    camera_coords_homogeneous = np.concatenate((camera_land_depth, np.tile(np.array([[1]]), (int(len(camera_land_depth)), 1))), axis=1)
    world_land = np.dot(camera_coords_homogeneous, c2w.T)
    landmarks = world_land[:, 0:3]
    return landmarks

def depth_compare(w2c_name:dict, K, dir_mesh:str, dir:str):
    """
    Compute depth error between reconstructed surface depth (before the projection) and GT depth map.
    The depth folder should be existed in parent path of dir
    :param w2c_name: Dict of poses and image names
    :param K: Intrinsic matrix
    :param dir_mesh: Path of the intraoperative model
    :param dir: Path of the GT depth images
    :return: Save results as pd file!
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import trimesh
    import OpenEXR
    import Imath
    import imageio.v2 as imageio
    from collections import OrderedDict
    import cv2
    import shutil
    import os
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2

    if isinstance(dir_mesh, str):
        source_mesh = trimesh.load(dir_mesh)
        source = [source_mesh.vertices]
    else:
        source = dir_mesh

    alpha_total_bg = []
    rmse_total_bg = []
    max_error_bg = []
    alpha_total_bg_spe = []
    rmse_total_bg_spe = []
    max_error_bg_spe = []

    remove_child_path = os.path.dirname(dir)
    depth_dir = os.path.join(remove_child_path, "depth")
    if not os.path.exists(depth_dir):
        raise "Can not find depth folder"
    f = os.listdir(depth_dir)
    depths = [depth for depth in sorted(f) if depth.endswith(".exr")]
    # Copy depth maps to other folder and synchronize file names with image names(frame_00000.png)
    new_depth_path = os.path.join(dir, "modified_depth")
    if not os.path.exists(new_depth_path):
        os.makedirs(new_depth_path)
    # Get depth names from mask_val file
    mask_val_path = os.path.join(dir, "mask_val_ERODE")
    assert os.path.exists(mask_val_path)
    mask_val_files = sorted(os.listdir(mask_val_path))
    H, W, _= cv2.imread(os.path.join(mask_val_path, mask_val_files[0])).shape

    for i, filename in enumerate(depths, start=1):
        # copy resized gt_depth(if there are not in the same size with "mask_val_ERODE"
        source_path = os.path.join(depth_dir, filename)
        modified_filename = f"frame_{i:05d}.exr"
        destination_path = os.path.join(new_depth_path, modified_filename)
        exr_depth = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)
        # gt_depth = get_exr_rgb(source_path) #read exr gt_depth
        if exr_depth.shape[0]!= H and exr_depth.shape[1] != W:
            depth_resize = cv2.resize(exr_depth, [W, H])
            cv2.imwrite(destination_path, depth_resize)
        # Copy the depth to the new folder
        # shutil.copy(source_path, destination_path)
    # resize the Camera intrinsic and get image coords if gt_depth is not the same size
    if exr_depth.shape[0] != H and exr_depth.shape[1] != W:
        K[0:2, :] = K[0:2, :] / (exr_depth.shape[0] / H)
    address = []
    names = []
    for member in mask_val_files: address.append(os.path.join("images", member))
    for dictionary in w2c_name:
        if dictionary["name"] in address:
            pose = np.array(dictionary["frame"])
            tumor = np.concatenate((source[0], np.tile(np.array([[1]]), (int(len(source[0])), 1))), axis=1)
            camera_coordinates = np.dot(tumor, pose.T)
            image_coordinates = np.dot(camera_coordinates, K.T)
            # Normalize point clouds (divid by z axis_depth)
            uv_z = image_coordinates / image_coordinates[:, 2:3]

            gt_dir = os.path.relpath(dictionary["name"], 'images')
            #Extract interesting coord on the bg masked maps
            extention_png = gt_dir + ".png"
            readmask_double_png = cv2.imread(os.path.join(dir, "masks", extention_png))[:,:, 0]
            if readmask_double_png.shape[0] != H and readmask_double_png.shape[1] != W:
                readmask_double_png = cv2.resize(readmask_double_png, [W, H])

            mask_pixels_bg = np.where(readmask_double_png > 0.0)
            mask_pixels_stack_bg = np.stack((mask_pixels_bg[1], mask_pixels_bg[0]), axis=1)

            # Extract interesting coord on the bg_spe masked maps
            readmask_bg_spe = cv2.imread(os.path.join(mask_val_path, gt_dir))[:,:, 0]
            mask_pixels_bg_sp = np.where(readmask_bg_spe > 0.0)
            mask_pixels_stack_bg_spe = np.stack((mask_pixels_bg_sp[1], mask_pixels_bg_sp[0]), axis=1)
            # Control whether the point clouds inside the frame or not
            # h, w = readmask_double_png.shape[0], readmask_double_png.shape[1]
            total_true = uv([uv_z[:, 0:2].astype(int)], W, H)
            xx = uv_z[:, 0:1][total_true[0]]
            yy = uv_z[:, 1:2][total_true[0]]
            zz = uv_z[:, 2:3][total_true[0]]
            uv_nerf = np.round(np.stack((xx, yy, zz), axis=1)).astype(int) # (n,3)
            # take into account only unique coordinates(they duplicated when convert float to int)
            # uv_unique_set = set(map(tuple, uv_nerf))
            # uv_unique_list = np.stack(list(uv_unique_set))[:, 0:2]

            # Do not assume unique coordinates
            # uv_unique_list = uv_nerf[:, 0:2]
            # sam = np.stack((xx, yy, zz), axis=1)

            # Control whether the point clouds are in the mask coordinates or not (wo_bg and wo_bg_spe)
            mask_tuple_bg = set(map(tuple, mask_pixels_stack_bg))
            mask_tuple_bg_spe = set(map(tuple, mask_pixels_stack_bg_spe))

            # Find the indices of the uniq coordinates in the uv_nerf that are in the mask_tuple_bg and mask_tuple_bg_spe
            selected_indices_bg = np.array(
                [idx for idx, coord in enumerate(uv_nerf[:, 0:2]) if tuple(coord) in mask_tuple_bg])

            selected_indices_bg_spe = np.array(
                [idx_sp for idx_sp, coord_sp in enumerate(uv_nerf[:, 0:2]) if tuple(coord_sp) in mask_tuple_bg_spe])

            selected_coordinates_bg = (uv_nerf[:, 0:2])[selected_indices_bg]
            selected_coordinates_bg_spe = (uv_nerf[:, 0:2])[selected_indices_bg_spe]

            # plot(sam[:, 0], sam[:,1],"black")
            # plot(uv_z[:, 0], uv_z[:, 1], "black")

            gt_depth_exr = gt_dir.split(".")[0] + ".exr"
            gt_depth = cv2.imread(os.path.join(new_depth_path, gt_depth_exr), cv2.IMREAD_UNCHANGED)[:, :, 0]

            selected_GT_bg = np.stack([gt_depth[y, x] for x, y in selected_coordinates_bg[:, 0:2]]) # GT depth wo_bg
            selected_GT_bg_spe = np.stack([gt_depth[y, x] for x, y in selected_coordinates_bg_spe[:, 0:2].astype(int)]) # GT depth wo_bg_spe
            # sam = gt_depth[mask_bg]
            # selected_GT_bg = gt_depth[readmask_double_png.astype(np.bool8)]
            # selected_GT_bg_spe = gt_depth[readmask_bg_spe.astype(np.bool8)]

            selected_rec_bg = image_coordinates[:, 2][selected_indices_bg] # reconstructed surface depth wo_bg
            selected_rec_bg_spe = image_coordinates[:, 2][selected_indices_bg_spe] # reconstructed surface depth wo_bg_spe

            alpha_bg = np.dot(np.transpose(selected_rec_bg), selected_GT_bg) / np.dot(np.transpose(selected_rec_bg), selected_rec_bg)
            alpha_bg_spe = np.dot(np.transpose(selected_rec_bg_spe), selected_GT_bg_spe) / np.dot(np.transpose(selected_rec_bg_spe), selected_rec_bg_spe)

            alpha_total_bg.append(alpha_bg)
            alpha_total_bg_spe.append(alpha_bg_spe)

            rmse_total_bg.append(rmse(selected_GT_bg, alpha_bg * selected_rec_bg) * 1000) # convert to mm
            rmse_total_bg_spe.append(rmse(selected_GT_bg_spe, alpha_bg_spe * selected_rec_bg_spe) * 1000) # convert to mm

            #Get root max square error for each frame
            max_error_bg.append(np.sqrt(np.square(np.subtract(selected_GT_bg, alpha_bg * selected_rec_bg)).max()) * 1000)  #in mm
            max_error_bg_spe.append(np.sqrt(np.square(np.subtract(selected_GT_bg_spe, alpha_bg_spe * selected_rec_bg_spe)).max()) * 1000)  #in mm
            names.append(gt_dir)
    d = {"Depth_wo_bg": rmse_total_bg, "Alpha_wo_bg": alpha_total_bg, "Max Error_wo_bg": max_error_bg,
         "Depth_wo_bg_spe": rmse_total_bg_spe, "Alpha_wo_bg_spe": alpha_total_bg_spe, "Max Error_wo_bg_spe": max_error_bg_spe, "Frame_Name": names}
    dd = pd.DataFrame(d)
    # add_std = pd.DataFrame({"RMSE":[dd["RMSE"].std(), dd["RMSE"].mean()], "Alpha":[dd["Alpha"].std(), dd["Alpha"].mean()],
    #                         "Max Error": [dd["Max Error"].std(), dd["Max Error"].mean()]})
    plus_minus = chr(0x00B1)
    add_std = pd.DataFrame(
        {"Depth_wo_bg": [str(np.round(dd["Depth_wo_bg"].mean(),2))+ plus_minus+ str(np.round(dd["Depth_wo_bg"].std(), 2))],
         "Alpha_wo_bg": [str(np.round(dd["Alpha_wo_bg"].mean(),2))+ plus_minus+ str(np.round(dd["Alpha_wo_bg"].std(), 3))],
         "Max Error_wo_bg": [str(np.round(dd["Max Error_wo_bg"].mean(), 2)) + plus_minus + str(np.round(dd["Max Error_wo_bg"].std(),2))],

         "Depth_wo_bg_spe": [str(np.round(dd["Depth_wo_bg_spe"].mean(), 2)) + plus_minus + str(np.round(dd["Depth_wo_bg_spe"].std(), 2))],
         "Alpha_wo_bg_spe": [str(np.round(dd["Alpha_wo_bg_spe"].mean(), 2)) + plus_minus + str(np.round(dd["Alpha_wo_bg_spe"].std(), 3))],
         "Max Error_wo_bg_spe": [str(np.round(dd["Max Error_wo_bg_spe"].mean(), 2)) + plus_minus + str(np.round(dd["Max Error_wo_bg_spe"].std(), 2))]
         })

    sam = dd.append(add_std, ignore_index= True)
    add_number = list(range(1, len(sam["Alpha_wo_bg"])))
    add_str = ["mean"+plus_minus+"std"]
    combined_values = add_number + add_str
    sam.insert(0, 'Numbers', combined_values)
    sam.to_csv(os.path.join(dir, "depth_result11.ods"))
    print("Excel Result saved in the folder!")
    return ()

def uv(UV, w, h):
    """Prepare coordinates before plotting
    Args:
        UV: list of coordinates(x,y)
        w: weight
        h: height
    Returns:
        true_list: list of prepared coords
    """
    true_total = []
    for s in range(len(UV)):
        u_true1 = UV[s][:, 0:1] < w-1
        u_true2 = UV[s][:, 0:1] >= 0.0
        u_true = u_true1 & u_true2
        v_true1 = UV[s][:, 1:2] < h-1
        v_true2 = UV[s][:, 1:2] >= 0.0
        v_true = v_true1 & v_true2
        true = u_true & v_true
        true_total.append(true)
    return(true_total)

def nerfacto2opencv(poses_dict:dict)->dict:
    """ convert Nerfacto(OpenGL) poses(x,y,z) to the Opencv system(x,-y,-z) then change
    "c2w" into "w2c".
    :param poses_dict: poses in OpenGL system --> Dict
    :return: posewithname: world to camera in OpenCV system with image names
    """
    import numpy as np

    posewithname = []
    gather = []
    for subject in poses_dict: gather.append(subject["frame"])
    poses = np.stack(gather)
    origins = poses[..., :3, 3]
    mean_origin = np.mean(origins, axis=0)
    translation = mean_origin
    # Align translation column with z axis
    up = np.mean(poses[:, :3, 1], axis=0)
    up = up / np.linalg.norm(up)
    rotation = rotation_matrix(up, np.array([0, 0, 1]))
    transform = np.concatenate([rotation, rotation @ -translation[..., None]], axis=-1)
    oriented_poses = transform @ poses
    poses = oriented_poses
    scale_factor = 1.0
    # poses[:, 0, 3] += 1.0  # we added +1 in the x axis of translation column
    scale_factor /= float(np.max(np.abs(poses[:, :3, 3])))
    poses[:, :3, 3] *= scale_factor
    # Convert from COLMAP's camera coordinate system to ours
    poses = np.concatenate((poses, np.tile(np.array([[0, 0, 0, 1]]), (int(len(poses)), 1, 1))), 1)
    poses[:, 0:3, 1:3] *= -1
    w2c = np.linalg.inv(poses)  # Change c2w to w2c
    #Add image name into W2C
    for key, value in zip(w2c, poses_dict):
        posewithname.append({"frame": key,"name":value["name"]})
    return posewithname

def readjson(dir:str):
    """
    Read json file and extract data
    poses are extracted along corresponding image names
    Parameters:
         dir: json file path
    Returns:
        frame_name_total: Global path of each image(all dataset images).
        K: intrinsic matrix .
        distortion: Distortion parameters as a vector .
        posewithname: Dict {poses(4*4) and name of the images}.
    """
    import numpy as np
    import json
    import os
    transform = open(os.path.join(dir, "transforms.json"))
    transforms = json.load(transform)
    # #Get distortion parameters
    if "w" not in transforms:
        total_transforms = transforms["frames"][0]
    else:
        total_transforms = transforms
    distortion_params = dict(
        k1=total_transforms["k1"] if "k1" in total_transforms else 0.0,
        k2=total_transforms["k2"] if "k2" in total_transforms else 0.0,
        k3=total_transforms["k3"] if "k3" in total_transforms else 0.0,
        k4=total_transforms["k4"] if "k4" in total_transforms else 0.0,
        p1=total_transforms["p1"] if "p1" in total_transforms else 0.0,
        p2=total_transforms["p2"] if "p2" in total_transforms else 0.0,
    )
    distortion_array = np.array([distortion_params["k1"], distortion_params["k2"], distortion_params["k3"],
                                distortion_params["k4"], distortion_params["p1"], distortion_params["p2"]])
    distortion = np.array([distortion_array[0], distortion_array[1], distortion_array[4], distortion_array[5], distortion_array[2]])
    # K matrix
    K = np.zeros((3, 4))
    K[0, 0] = total_transforms["fl_x"]
    K[0, 2] = total_transforms["cx"]
    K[1, 1] = total_transforms["fl_y"]
    K[1, 2] = total_transforms["cy"]
    K[2, 2] = 1
    # Generate rgb paths
    fnames = []
    for frame in transforms["frames"]:
        filepath = frame["file_path"]
        fnames.append(os.path.join(dir, filepath))
    inds = np.argsort(fnames)
    frames = [transforms["frames"][ind] for ind in inds]
    frame_name_total = []
    posewithname= []
    for framee in frames:
        frame_pose = framee["transform_matrix"]
        frame_name = framee["file_path"]
        # frame_pose_total.append(frame_pose)
        frame_name_total.append(os.path.join(dir,frame_name))
        posewithname.append({"frame": frame_pose, "name": frame_name})
    return(frame_name_total, K, distortion, posewithname)


def get_exr_rgb(path):
    import OpenEXR
    import Imath
    import numpy as np

    I = OpenEXR.InputFile(path)
    dw = I.header()['displayWindow']
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    data = [np.fromstring(c, np.float32).reshape(size) for c in I.channels("RGB", Imath.PixelType(Imath.PixelType.FLOAT))]
    img = np.dstack(data)
    return img


# def chamfer_distance(points_a, points_b):
#     import numpy as np
#     # Compute the distances from each point in points_a to the nearest point in points_b
#     dist_a_to_b = np.min(np.linalg.norm(source[:, np.newaxis] - target, axis=2), axis=1)
#     # Compute the distances from each point in points_b to the nearest point in points_a
#     dist_b_to_a = np.min(np.linalg.norm(points_b[:, np.newaxis] - points_a, axis=2), axis=1)
#
#     # Chamfer distance is the sum of the averages of these squared distances
#     chamfer_dist = np.mean(dist_a_to_b) + np.mean(dist_b_to_a)
#     return chamfer_dist

if __name__ == "__main__":
    icp_function()
