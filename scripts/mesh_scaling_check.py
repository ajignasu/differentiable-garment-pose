import matplotlib.pyplot as plt
import numpy as np
import trimesh


# load mesh1 and mesh2
mesh1 = trimesh.load_mesh('D:/Code/differentiable_rendering/data/tshirt_scaled.obj')
mesh2 = trimesh.load_mesh('D:/Code/differentiable_rendering/data/smpl_default.obj')


# range of mesh1
print('range of mesh1')
print(mesh1.bounds)
print('range of mesh2')
print(mesh2.bounds)

# normalize mesh1 to fit in [-1, 1]
mesh1.vertices -= mesh1.bounds.mean(axis=0)
mesh1.vertices /= np.abs(mesh1.vertices).max()
# mesh1.vertices *= 0.125

# normalize mesh2 to fit in [-1, 1]
mesh2.vertices -= mesh2.bounds.mean(axis=0)
mesh2.vertices /= np.abs(mesh2.vertices).max()

print('range of mesh1')
print(mesh1.bounds)

print('range of mesh2')
print(mesh2.bounds)


# # plot mesh1 and mesh2 in 3D using proper bounds
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(mesh1.bounds[:, 0])
ax.set_ylim(mesh1.bounds[:, 1])
ax.set_zlim(mesh1.bounds[:, 2])
ax.plot_trisurf(mesh1.vertices[:, 0], mesh1.vertices[:, 1], mesh1.vertices[:, 2], triangles=mesh1.faces, color='b')
ax.plot_trisurf(mesh2.vertices[:, 0], mesh2.vertices[:, 1], mesh2.vertices[:, 2], triangles=mesh2.faces, color='r')
plt.show()


