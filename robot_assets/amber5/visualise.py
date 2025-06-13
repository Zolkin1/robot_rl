import os
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Parse URDF
urdf_path = 'amber.urdf'
tree = ET.parse(urdf_path)
root = tree.getroot()

# Function to create homogeneous transform from xyz and rpy
def transform_matrix(xyz, rpy):
    from math import cos, sin
    r, p, y = rpy
    Rx = np.array([[1, 0, 0],
                   [0, cos(r), -sin(r)],
                   [0, sin(r), cos(r)]])
    Ry = np.array([[cos(p), 0, sin(p)],
                   [0, 1, 0],
                   [-sin(p), 0, cos(p)]])
    Rz = np.array([[cos(y), -sin(y), 0],
                   [sin(y), cos(y), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T

# Extract visuals with mesh geometry
link_visuals = {}
for link in root.findall('link'):
    name = link.get('name')
    visuals = []
    for vis in link.findall('visual'):
        geom = vis.find('geometry/mesh')
        if geom is None:
            continue
        filename = geom.get('filename')
        scale_str = geom.get('scale') or "1 1 1"
        scale = list(map(float, scale_str.split()))
        origin_elem = vis.find('origin')
        if origin_elem is not None:
            xyz_attr = origin_elem.get('xyz')
            rpy_attr = origin_elem.get('rpy')
            xyz = list(map(float, xyz_attr.split())) if xyz_attr else [0, 0, 0]
            rpy = list(map(float, rpy_attr.split())) if rpy_attr else [0, 0, 0]
        else:
            xyz = [0, 0, 0]
            rpy = [0, 0, 0]
        visuals.append({'filename': filename, 'scale': scale, 'xyz': xyz, 'rpy': rpy})
    if visuals:
        link_visuals[name] = visuals

# Extract joint transforms
joints = []
for joint in root.findall('joint'):
    parent = joint.find('parent').get('link')
    child = joint.find('child').get('link')
    origin_elem = joint.find('origin')
    if origin_elem is not None:
        xyz_attr = origin_elem.get('xyz')
        rpy_attr = origin_elem.get('rpy')
        xyz = list(map(float, xyz_attr.split())) if xyz_attr else [0, 0, 0]
        rpy = list(map(float, rpy_attr.split())) if rpy_attr else [0, 0, 0]
    else:
        xyz = [0, 0, 0]
        rpy = [0, 0, 0]
    joints.append({'parent': parent, 'child': child, 'xyz': xyz, 'rpy': rpy})

# Build global link transforms starting from 'world'
root_joint = next((j for j in joints if j['parent'] == 'world'), None)
root_link = root_joint['child']
link_transforms = {root_link: transform_matrix(root_joint['xyz'], root_joint['rpy'])}

# Propagate transforms
while True:
    progressed = False
    for j in joints:
        parent, child = j['parent'], j['child']
        if parent in link_transforms and child not in link_transforms:
            Tj = transform_matrix(j['xyz'], j['rpy'])
            link_transforms[child] = link_transforms[parent] @ Tj
            progressed = True
    if not progressed:
        break

# Load and transform meshes
meshes = []
for link_name, visuals in link_visuals.items():
    if link_name not in link_transforms:
        continue
    T_link = link_transforms[link_name]
    for vis in visuals:
        mesh_fname = vis['filename']
        mesh_path = os.path.join('', mesh_fname)
        if not os.path.exists(mesh_path):
            mesh_path = os.path.join('', os.path.basename(mesh_fname))
        mesh = trimesh.load(mesh_path, force='mesh')
        mesh.apply_scale(vis['scale'])
        T_vis = transform_matrix(vis['xyz'], vis['rpy'])
        mesh.apply_transform(T_vis)
        mesh.apply_transform(T_link)
        meshes.append(mesh)

# Plot using matplotlib
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for mesh in meshes:
    v = mesh.vertices
    f = mesh.faces
    # Create a Poly3DCollection
    triangles = v[f]
    poly = Poly3DCollection(triangles, alpha=0.7, edgecolor='k')
    ax.add_collection3d(poly)

# Set axis limits
all_verts = np.vstack([mesh.vertices for mesh in meshes])
mins = all_verts.min(axis=0)
maxs = all_verts.max(axis=0)
ax.set_xlim(mins[0], maxs[0])
ax.set_ylim(mins[1], maxs[1])
ax.set_zlim(mins[2], maxs[2])

ax.set_box_aspect((maxs - mins))
plt.axis('off')
plt.show()
