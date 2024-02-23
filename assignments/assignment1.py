import trimesh
import numpy as np

from collections import defaultdict
from functools import reduce


# [START]: LOOP SUBDIVISION IMPLEMENTATION
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
            )  # get axes-wise sum of all vertices with both vertices in the face

    if n_faces == 2:  # edge is the interior
        adjacent_sum += vertices[v_idx_0] + vertices[v_idx_1]  # add the vertices t0 the axes-wise sum once more
        return adjacent_sum / 8.  # at this point all vertices are in the correct proportion, simply divide by 8
    else:  # edge is on the boundary
        return (vertices[v_idx_0] + vertices[v_idx_1]) / 2.  # simply return the center of the edge


def _calculate_odd_vertices(vertices, faces):
    odd_vertices = dict()

    for face in faces:
        for i in range(len(face)):
            edge = (face[i], face[(i + 1) % len(face)])  # get all edges from current face
            if edge not in odd_vertices:
                odd_vertices[edge] = _calculate_odd_vertex(*edge, faces, vertices)  # new vertex for this edge
                odd_vertices[edge[::-1]] = odd_vertices[edge]  # add a duplicate (e1, e0) entry for every (e0, e1) edge

    return odd_vertices


def _calculate_even_vertices(vertices, faces):
    even_vertices = dict()

    for idx, vertex in enumerate(vertices):
        adjacent_vertices = list(set(reduce(
            lambda acc, face: acc + [v for v in face if v != idx],
            filter(lambda face: idx in face, faces),
            list()
        )))  # get a list of all vertices adjacent to the current vertex

        if 0 <= len(adjacent_vertices) < 2:
            even_vertex = vertex
        else:
            sum_adjacent = reduce(
                lambda acc, v: acc + vertices[v],
                adjacent_vertices,
                np.zeros_like(vertex)
            )  # sum the adjacent vertices axis-wise
            if len(adjacent_vertices) == 2:  # interior edge
                even_vertex = (1. / 8.) * sum_adjacent + (3. / 4.) * vertex  # use the formula on slide 60
            else:  # boundary edge
                k = len(adjacent_vertices)
                beta = (5. / 8. - (3. / 8. + 1. / 4. * np.cos(2 * np.pi / k)) ** 2) / k  # use the formula on slide 60
                even_vertex = vertex * (1 - k * beta) + beta * sum_adjacent

        even_vertices[idx] = even_vertex  # add an entry for the new even vertex corresponding to the old vertex

    return even_vertices


def _compose_new_faces(odd_vertices, even_vertices, faces):
    new_vertices = list(set(map(
        lambda vertex: (vertex[0], vertex[1], vertex[2]),
        list(odd_vertices.values()) + list(even_vertices.values())
    )))  # combine the bew odd and even vertices
    new_faces = list()

    new_vertices_dict = dict(map(
        lambda idx_vertex: (idx_vertex[1], idx_vertex[0]),
        enumerate(new_vertices)
    ))  # create a dictionary form the combined list for fast lookup

    for face in faces:
        new_faces.extend([  # for every face, add the four new faces
            [  # face 1 with one even vertex and two odd vertices
                new_vertices_dict[tuple(even_vertices[face[0]])],
                new_vertices_dict[tuple(odd_vertices[(face[0], face[1])])],
                new_vertices_dict[tuple(odd_vertices[(face[2], face[0])])],
            ],
            [  # face 2 with one even vertex and two odd vertices
                new_vertices_dict[tuple(even_vertices[face[1]])],
                new_vertices_dict[tuple(odd_vertices[(face[0], face[1])])],
                new_vertices_dict[tuple(odd_vertices[(face[1], face[2])])],
            ],
            [  # face 3 with one even vertex and two odd vertices
                new_vertices_dict[tuple(even_vertices[face[2]])],
                new_vertices_dict[tuple(odd_vertices[(face[1], face[2])])],
                new_vertices_dict[tuple(odd_vertices[(face[2], face[0])])],
            ],
            [  # interior face (4) with all odd vertices
                new_vertices_dict[tuple(odd_vertices[(face[0], face[1])])],
                new_vertices_dict[tuple(odd_vertices[(face[1], face[2])])],
                new_vertices_dict[tuple(odd_vertices[(face[2], face[0])])],
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

    # reference 1: https://www.cs.cmu.edu/afs/cs/academic/class/15462-s14/www/lec_slides/Subdivision.pdf
    # reference 2: https://www.dropbox.com/scl/fo/r8ktikl0qmuqs4cuk7f8x/h?dl=0&e=1&preview=Geometry_processing.pdf&rlkey=ym40kcihfcfydyxd4wat0806g

    if iterations < 1:
        return mesh

    for _ in range(iterations):
        # step 1: get vertices and faces from the mesh
        vertices, faces = mesh.vertices, mesh.faces
        # step 2: generate odd vertices from the edges
        odd_vertices = _calculate_odd_vertices(vertices, faces)
        # step 3: generate even vertices from the original vertices
        even_vertices = _calculate_even_vertices(vertices, faces)
        # step 4: create a new mesh
        mesh = trimesh.Trimesh(*_compose_new_faces(odd_vertices, even_vertices, faces))

    return mesh
# [END]: LOOP SUBDIVISION IMPLEMENTATION


# [START]: QUADRATIC ERROR DECIMATION IMPLEMENTATION
def _get_vertex_quadrics(vertices, faces):
    vertex_quadrics = defaultdict(lambda: np.zeros((4, 4)))  # mapping from vertex index to its quadrics matrix

    for face in faces:
        # for the current triangle (face), find the quadrics matrix K
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        normal = np.cross(v1 - v0, v2 - v0)
        normal /= np.linalg.norm(normal)
        a, b, c = normal
        d = -np.dot(normal, v0)

        # create the K (quadric) matrix (slide 68)
        plane_matrix = np.array([[a * a, a * b, a * c, a * d],
                                 [a * b, b * b, b * c, b * d],
                                 [a * c, b * c, c * c, c * d],
                                 [a * d, b * d, c * d, d * d]])

        # accumulate K to all vertices of the current mesh
        vertex_quadrics[face[0]] += plane_matrix
        vertex_quadrics[face[1]] += plane_matrix
        vertex_quadrics[face[2]] += plane_matrix

    return vertex_quadrics


def _get_edge_quadrics(faces, vertex_quadrics):
    edge_quadrics = dict()  # mapping from an edge to the quadrics matrix

    for face in faces:
        for i in range(len(face)):
            # for every edge in the mesh, combine the K matrices and edd entry
            edge = (i, (i + 1) % len(face))
            edge_quadrics[edge] = vertex_quadrics[edge[0]] + vertex_quadrics[edge[1]]

    return edge_quadrics


def _get_min_cost_edge(edge_quadrics):
    min_error = float('inf')  # AKA cost
    min_edge = (-1, -1)
    min_pt = (np.nan, np.nan)

    for edge, Kij in edge_quadrics.items():
        # decompose K (slide 73)
        B = Kij[:3, :3]
        w = Kij[:3, 3]
        # d_sq = Kij[3, 3]

        # find error and minimum point
        pt = -np.linalg.inv(B) @ w
        pt_4d = np.append(pt, 1)
        error = pt_4d.T @ Kij @ pt_4d

        # update the minimum cost edge
        if error < min_error:
            min_error = error
            min_pt = pt
            min_edge = edge

    return min_edge, min_pt


def _collapse_edge(vertices, faces, min_edge, min_pt):
    new_vertices = np.vstack((np.array(list(map(
        lambda v_idx: vertices[v_idx],
        filter(lambda v_idx: v_idx not in min_edge, range(len(vertices)))
    ))), min_pt))  # filter out vertices to be removed and add new vertex for the collapsed edge
    new_vertices_dict = dict(map(
        lambda v_idx: (tuple(v_idx[1]), v_idx[0]),
        enumerate(new_vertices)
    ))  # create a dictionary for fast lookups

    # create new faces from the new and old vertices
    new_faces = np.array(list(map(
        lambda face: (
            new_vertices_dict[tuple(vertices[face[0]]) if face[0] != '*' else tuple(min_pt)],
            new_vertices_dict[tuple(vertices[face[1]]) if face[1] != '*' else tuple(min_pt)],
            new_vertices_dict[tuple(vertices[face[2]]) if face[2] != '*' else tuple(min_pt)],
        ),  # step 4. update the indices of the vertices with indices from the new vertex list and replace the * symbol
        map(
            lambda face: (
                face[0] if face[0] not in min_edge else '*',
                face[1] if face[1] not in min_edge else '*',
                face[2] if face[2] not in min_edge else '*',
            ),  # step 3. if a vertex is in min edge replace it with a special symbol *
            filter(
                lambda face: not (min_edge[0] in face and min_edge[1] in face),
                map(tuple, faces)  # step 1. map face vectors in faces to a tuple (that can be used for lookup)
            )  # step 2. filter out the faces that contain both vertices of the edge to be removed
        )
    )))

    return new_vertices, new_faces


def simplify_quadratic_error(mesh, face_count=1):
    """
    Apply quadratic error mesh decimation to the input mesh until the target face count is reached.
    :param mesh: input mesh
    :param face_count: number of faces desired in the resulting mesh.
    :return: mesh after decimation
    """

    # reference: slide 67 to 74
    # https://www.dropbox.com/scl/fo/r8ktikl0qmuqs4cuk7f8x/h?dl=0&e=1&preview=Geometry_processing.pdf&rlkey=ym40kcihfcfydyxd4wat0806g

    vertices, faces = mesh.vertices, mesh.faces
    if face_count >= mesh.faces.size // 3:
        return mesh

    while faces.size // 3 > face_count:
        # step 1: calculate quadric matrices for every vertex and edge
        vertex_quadrics = _get_vertex_quadrics(vertices, faces)
        edge_quadrics = _get_edge_quadrics(faces, vertex_quadrics)
        # step 2: find the edge with the minimum cost (the least error) and get the corresponding new vertex
        min_edge, min_pt = _get_min_cost_edge(edge_quadrics)
        # step 3: collapse the edge to the new vertex and recompute the vertices and faces
        vertices, faces = _collapse_edge(vertices, faces, min_edge, min_pt)

    return trimesh.Trimesh(vertices, faces)
# [END]: QUADRATIC ERROR DECIMATION IMPLEMENTATION


if __name__ == '__main__':
    # Load mesh and print information
    # object_mesh = trimesh.load_mesh('assets/cube.obj')
    object_mesh = trimesh.creation.box(extents=[0.5, 0.5, 0.5])
    print(f'Mesh Info: {object_mesh}')

    # PART A) GROUND TRUTH - loop subdivision from trimesh
    # mesh_subdivided = trimesh.Trimesh(*trimesh.remesh.subdivide_loop(object_mesh.vertices, object_mesh.faces, 1))

    # PART A) IMPLEMENTATION - implement your own loop subdivision here
    mesh_subdivided = subdivision_loop(object_mesh, iterations=1)

    # print the new mesh information and save the mesh
    print(f'Subdivided Mesh Info: {mesh_subdivided}')
    mesh_subdivided.export('assets/assignment1/cube_subdivided.obj')

    # # PART B) GROUND TRUTH - quadratic error mesh decimation
    # mesh_decimated = object_mesh.simplify_quadric_decimation(4)

    # PART B) IMPLEMENTATION - implement your own quadratic error mesh decimation here
    mesh_decimated = simplify_quadratic_error(object_mesh, face_count=4)

    # print the new mesh information and save the mesh
    print(f'Decimated Mesh Info: {mesh_decimated}')
    mesh_decimated.export('assets/assignment1/cube_decimated.obj')
