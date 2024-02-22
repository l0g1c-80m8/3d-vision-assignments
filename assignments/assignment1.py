import trimesh
import numpy as np
from functools import reduce


def _calculate_odd_vertex(v_idx_0, v_idx_1, faces, vertices):
    n_faces = 0
    adjacent_sum = np.zeros_like(vertices[0])

    for face in faces:
        if v_idx_0 in face and v_idx_1 in face:
            n_faces += 1
            adjacent_sum = reduce(
                lambda acc, v_idx: acc + vertices[v_idx],
                face,
                adjacent_sum
            )

    if n_faces == 2:
        adjacent_sum += vertices[v_idx_0] + vertices[v_idx_1]
        return adjacent_sum / 8.
    else:
        return (vertices[v_idx_0] + vertices[v_idx_1]) / 2.


def _calculate_odd_vertices(vertices, faces):
    odd_vertices = dict()

    for face in faces:
        for i in range(len(face)):
            edge = (face[i], face[(i + 1) % len(face)])
            if edge not in odd_vertices:
                odd_vertices[edge] = _calculate_odd_vertex(*edge, faces, vertices)
                odd_vertices[edge[::-1]] = odd_vertices[edge]

    return odd_vertices


def _calculate_even_vertices(vertices, faces):
    even_vertices = dict()

    for idx, vertex in enumerate(vertices):
        adjacent_vertices = list(set(reduce(
            lambda acc, face: acc + [v for v in face if v != idx],
            filter(lambda face: idx in face, faces),
            list()
        )))

        if 0 <= len(adjacent_vertices) < 2:
            even_vertex = vertex
        else:
            sum_adjacent = reduce(
                lambda acc, v: acc + vertices[v],
                adjacent_vertices,
                np.zeros_like(vertex)
            )
            if len(adjacent_vertices) == 2:
                even_vertex = (1. / 8.) * sum_adjacent + (3. / 4.) * vertex
            else:
                k = len(adjacent_vertices)
                beta = (40. - (3 + 2 * np.cos(2 * np.pi / k)) ** 2) / (k * 64.)
                even_vertex = vertex * (1 - k * beta) + beta * sum_adjacent

        even_vertices[idx] = even_vertex

    return even_vertices


def _compose_new_faces(odd_vertices, even_vertices, faces):
    print(odd_vertices)
    print()
    print(even_vertices)
    new_vertices = list(map(
        lambda vertex: (vertex[0], vertex[1], vertex[2]),
        list(odd_vertices.values()) + list(even_vertices.values())
    ))
    print()
    print(new_vertices)
    new_faces = list()

    for face in faces:
        new_faces.extend([
            [
                new_vertices.index(tuple(even_vertices[face[0]])),
                new_vertices.index(tuple(odd_vertices[(face[0], face[1])])),
                new_vertices.index(tuple(odd_vertices[(face[0], face[2])])),
            ],
            [
                new_vertices.index(tuple(even_vertices[face[1]])),
                new_vertices.index(tuple(odd_vertices[(face[1], face[0])])),
                new_vertices.index(tuple(odd_vertices[(face[1], face[2])])),
            ],
            [
                new_vertices.index(tuple(even_vertices[face[2]])),
                new_vertices.index(tuple(odd_vertices[(face[2], face[0])])),
                new_vertices.index(tuple(odd_vertices[(face[2], face[1])])),
            ],
            [
                new_vertices.index(tuple(odd_vertices[(face[0], face[1])])),
                new_vertices.index(tuple(odd_vertices[(face[1], face[2])])),
                new_vertices.index(tuple(odd_vertices[(face[2], face[1])])),
            ],
        ])

    return new_vertices, new_faces


def subdivision_loop(mesh, iterations=1):
    """
    Apply Loop subdivision to the input mesh for the specified number of iterations.
    :param mesh: input mesh
    :param iterations: number of iterations
    :return: mesh after subdivision
    
    Overall process:
    Reference: https://github.com/mikedh/trimesh/blob/main/trimesh/remesh.py#L207
    1. Calculate odd vertices.
      Assign a new odd vertex on each edge and
      calculate the value for the boundary case and the interior case.
      The value is calculated as follows.
          v2
        / f0 \\        0
      v0--e--v1      /   \\
        \\f1 /     v0--e--v1
          v3
      - interior case : 3:1 ratio of mean(v0,v1) and mean(v2,v3)
      - boundary case : mean(v0,v1)
    2. Calculate even vertices.
      The new even vertices are calculated with the existing
      vertices and their adjacent vertices.
        1---2
       / \\/ \\      0---1
      0---v---3     / \\/ \\
       \\ /\\/    b0---v---b1
        k...4
      - interior case : (1-kB):B ratio of v and k adjacencies
      - boundary case : 3:1 ratio of v and mean(b0,b1)
    3. Compose new faces with new vertices.
    
    # The following implementation considers only the interior cases
    # You should also consider the boundary cases and more iterations in your submission
    """

    # reference: https://www.cs.cmu.edu/afs/cs/academic/class/15462-s14/www/lec_slides/Subdivision.pdf

    vertices, faces = mesh.vertices, mesh.faces

    for _ in range(iterations):
        odd_vertices = _calculate_odd_vertices(vertices, faces)
        even_vertices = _calculate_even_vertices(vertices, faces)
        vertices, faces = _compose_new_faces(odd_vertices, even_vertices, faces)

    return trimesh.Trimesh(vertices, faces)


def simplify_quadric_error(mesh, face_count=1):
    """
    Apply quadratic error mesh decimation to the input mesh until the target face count is reached.
    :param mesh: input mesh
    :param face_count: number of faces desired in the resulting mesh.
    :return: mesh after decimation
    """
    return mesh


if __name__ == '__main__':
    # Load mesh and print information
    # object_mesh = trimesh.load_mesh('assets/cube.obj')
    object_mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, -1, 0]],
                       faces=[[0, 1, 2], [0, 1, 3]])
    print(f'Mesh Info: {object_mesh}')

    # implement your own loop subdivision here
    mesh_subdivided = subdivision_loop(object_mesh)

    # print the new mesh information and save the mesh
    print(f'Subdivided Mesh Info: {mesh_subdivided}')
    mesh_subdivided.export('assets/assignment1/cube_subdivided.obj')

    # quadratic error mesh decimation
    # mesh_decimated = object_mesh.simplify_quadric_decimation(4)

    # TODO: implement your own quadratic error mesh decimation here
    # mesh_decimated = simplify_quadric_error(mesh, face_count=1)

    # print the new mesh information and save the mesh
    # print(f'Decimated Mesh Info: {mesh_decimated}')
    # mesh_decimated.export('assets/assignment1/cube_decimated.obj')
