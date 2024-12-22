from dataclasses import dataclass

import numpy as np

import maya.cmds as cmds
import maya.mel as mel

import pymel.core as pm
import pymel.core.datatypes as dt
import pymel.core.nodetypes as nt

from . import utils
from importlib import reload
reload(utils)
@dataclass
class CopyOptions:
    distance_threshold: float = 0.1
    angle_threshold: float = 10
    flip_vertex_normal: bool = True


@dataclass
class SmoothOptions:
    num_iter_step: int = 2
    alpha: float = 0.1
    range_distance: float = 1.0


@dataclass
class LimitOptions:
    bone_num: int = 4
    dilation_repeat: int = 5


class WeightTransfer(object):
    def __init__(self, source, target,
                 copy_options: CopyOptions,
                 smooth_options: SmoothOptions | None,
                 limit_options: LimitOptions | None):
        self._source = self.__get_mesh(source)
        self._target = self.__get_mesh(target)
        self._copy_options = copy_options
        self._smooth_options = smooth_options
        self._limit_options = limit_options

    @staticmethod
    def __get_mesh(obj: object) -> nt.Mesh:
        if isinstance(obj, str):
            obj = pm.PyNode(obj)
        elif isinstance(obj, pm.PyNode):
            pass
        else:
            raise TypeError(f'{obj} source must be str or pm.PyNode')

        if isinstance(obj, nt.Mesh):
            return obj
        elif isinstance(obj, nt.Transform) and (shape:= obj.getShape()) :
            if isinstance(shape, nt.Mesh):
                return shape
            else:
                raise TypeError(f'{obj} shape must be nt.Mesh or nt.Transform')
        else:
            raise TypeError(f'{obj} source must be nt.Mesh or nt.Transform')

    @staticmethod
    def __get_skin_node(mesh: nt.Mesh) -> list[nt.SkinCluster]:
        return [_ for _ in pm.findDeformers(mesh) or [] if isinstance(pm.PyNode(_), nt.SkinCluster)]

    def run(self):

        # region source data
        source_skin = self.__get_skin_node(self._source)
        assert source_skin, RuntimeError('source skin not found.')
        source_skin = source_skin[0]

        source_skin_bone = pm.skinCluster(source_skin, q=True, inf=True)
        source_vertices = np.empty((len(self._source.vtx), 3), dtype=np.float32)
        source_normals = np.empty((len(self._source.vtx), 3), dtype=np.float32)
        source_weights = np.empty((len(self._source.vtx), len(source_skin_bone)), dtype=np.float32)
        for vtx in self._source.vtx:
            vp = vtx.getPosition(dt.Space.kWorld)
            source_vertices[vtx.index()] = [vp.x, vp.y, vp.z]
            source_normals[vtx.index()] = vtx.getNormals()[0]
            source_weights[vtx.index()] = pm.skinPercent(source_skin, vtx, q=True, v=True)

        source_triangle_count = self._source.numTriangles()
        source_triangle_data = self._source.getTriangles()[1]
        source_triangles = np.empty((source_triangle_count, 3), dtype=np.int64)
        for fi in range(source_triangle_count):
            source_triangles[fi] = [source_triangle_data[fi * 3 + 0],
                                    source_triangle_data[fi * 3 + 1],
                                    source_triangle_data[fi * 3 + 2]
                                    ]
        # endregion

        # region target data
        target_vertices = np.empty((len(self._target.vtx), 3), dtype=np.float32)
        target_normals = np.empty((len(self._target.vtx), 3), dtype=np.float32)
        for vtx in self._target.vtx:
            vp = vtx.getPosition(dt.Space.kWorld)
            target_vertices[vtx.index()] = [vp.x, vp.y, vp.z]
            target_normals[vtx.index()] = vtx.getNormals()[0]

        target_triangle_count = self._target.numTriangles()
        target_triangle_data = self._target.getTriangles()[1]
        target_triangles = np.empty((target_triangle_count, 3), dtype=np.int64)

        # generate edge data
        target_edge_data = {}
        for fi in range(target_triangle_count):
            v1 = target_triangle_data[fi * 3 + 0]
            v2 = target_triangle_data[fi * 3 + 1]
            v3 = target_triangle_data[fi * 3 + 2]
            target_triangles[fi] = [v1, v2, v3]

            sorted_index = sorted([v1, v2, v3])
            s1, s2, s3 = sorted_index[0], sorted_index[1], sorted_index[2]
            if s1 not in target_edge_data:
                target_edge_data[s1] = []
            if [s1, s2] not in target_edge_data[s1]:
                target_edge_data[s1].append([s1, s2])
            if [s1, s3] not in target_edge_data[s1]:
                target_edge_data[s1].append([s1, s3])

            if s2 not in target_edge_data:
                target_edge_data[s2] = []
            if [s2, s3] not in target_edge_data[s2]:
                target_edge_data[s2].append([s2, s3])

        target_edge_result_data = [ed for ev in target_edge_data.values() for ed in ev]
        target_edge = np.empty((len(target_edge_result_data), 2), dtype=np.int64)
        for ei, e in enumerate(target_edge_result_data):
            target_edge[ei] = e

        # endregion

        # region calculate weight
        # copy weight
        matched_verts, matched_weights = utils.find_matches_closest_surface(
            source_vertices, source_triangles, source_normals,
            target_vertices, target_normals, source_weights,
            self._copy_options.distance_threshold,
            self._copy_options.angle_threshold,
            self._copy_options.flip_vertex_normal
        )
        # inpaint weight
        result, inpaint_weights = utils.inpaint(
            target_vertices, target_triangles, matched_weights, matched_verts, True
        )
        to_used_weight = inpaint_weights
        # smooth weight
        adjacency_matrix, adj_list = utils.get_adjacency_data(target_vertices, target_edge)
        if self._smooth_options:
            smooth_weights = utils.smooth_weigths(
                target_vertices, inpaint_weights, matched_verts, adjacency_matrix, adj_list,
                self._smooth_options.num_iter_step,
                self._smooth_options.alpha,
                self._smooth_options.range_distance
            )
            to_used_weight = smooth_weights
        # Limit the number of vertex weight influences
        if self._limit_options:
            to_used_weight[to_used_weight < 0.0] = 0.0
            mask = utils.limit_mask(to_used_weight, adjacency_matrix,
                                    self._limit_options.bone_num,
                                    self._limit_options.dilation_repeat
                                    )
            to_used_weight = (1 - mask) * to_used_weight
            to_used_weight[to_used_weight < 0.0] = 0.0

        # endregion

        # region set skin weight
        target_skin_node = self.__get_skin_node(self._target)
        if not target_skin_node: # not found skinCluster, create
            target_skin_node = pm.skinCluster(source_skin_bone, self._target, bm=0, sm=0)
        else:
            target_skin_node = target_skin_node[0]
            target_skin_bone = pm.skinCluster(target_skin_node, q=True, inf=True)
            source_skin_bone_name = [_.name() for _ in source_skin_bone]
            target_skin_bone_name = [_.name() for _ in target_skin_bone]
            if diff := list(set(source_skin_bone_name) - set(target_skin_bone_name)):
                # replenish bone influences
                pm.skinCluster(target_skin_node, e=True, ai=diff, lw=True, wt=0.0)
        # set bone weight
        for vtx in self._target.vtx:
            vtx_weight = to_used_weight[vtx.index()]
            vtx_weight[vtx_weight < 0.0] = 0.0
            tv = [[source_skin_bone[_i].name(), _] for _i, _ in enumerate(vtx_weight.astype(float))]
            pm.skinPercent(target_skin_node, vtx, tv=tv)
        # endregion
        return
