#include <pbrt/cpu/grid.h>

#include <pbrt/interaction.h>
#include <pbrt/paramdict.h>
#include <pbrt/shapes.h>
#include <pbrt/util/error.h>
#include <pbrt/util/log.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/stats.h>

namespace pbrt {

GridAggregate::GridAggregate(std::vector<Primitive> prims)
    : primitives(std::move(prims)){
    for (uint32_t i = 0; i < prims.size(); ++i)
        bounds = Union(bounds, primitives[i].Bounds());
    Vector3f delta = bounds.pMax - bounds.pMin;

    int maxAxis = bounds.MaxDimension();
    float invMaxWidth = 1.0f / delta[maxAxis];

    float cubeRoot = 3.0f * std::pow(float(primitives.size()), 1.0f / 3.0f);
    float voxelsPerUnitDist = cubeRoot * invMaxWidth;

    for (int axis = 0; axis < 3; ++axis) {
        nVoxels[axis] = (int)std::round(delta[axis] * voxelsPerUnitDist);
        nVoxels[axis] = Clamp(nVoxels[axis], 1, 64);
    }

    for (int axis = 0; axis < 3; ++axis) {
        width[axis] = delta[axis] / nVoxels[axis];
        invWidth[axis] = (width[axis] == 0.f) ? 0.f : 1.f / width[axis];
    }

    int nv = nVoxels[0] * nVoxels[1] * nVoxels[2];
    voxels.resize(nv, std::vector<int>());

    for (int i = 0; i < primitives.size(); ++i) {
        Bounds3f pb = primitives[i].Bounds();

        int vmin[3], vmax[3];
        for (int axis = 0; axis < 3; ++axis) {
            vmin[axis] = posToVoxel(pb.pMin, axis);
            vmax[axis] = posToVoxel(pb.pMax, axis);
        }
        for (int z = vmin[2]; z <= vmax[2]; ++z)
            for (int y = vmin[1]; y <= vmax[1]; ++y)
                for (int x = vmin[0]; x <= vmax[0]; ++x) {
                    int o = offset(x, y, z);
                    voxels[o].push_back(i);
                }


    }
}

pstd::optional<ShapeIntersection> GridAggregate::Intersect(const Ray &r, Float tMax) const {
    Ray ray = r;
    Float rayT, tMinBounds, tMaxBounds;

    if (!bounds.IntersectP(ray.o, ray.d, tMax, &tMinBounds, &tMaxBounds)){
        return {};
    }
    rayT = tMinBounds;
    Point3f gridIntersect = ray(rayT);

    float NextCrossingT[3], DeltaT[3];
    int Step[3], Out[3], Pos[3];
    for (int axis = 0; axis < 3; ++axis) {
        Pos[axis] = posToVoxel(gridIntersect, axis);
        if (ray.d[axis] == -0.f) ray.d[axis] = 0.f;

        if (ray.d[axis] >= 0) {
            NextCrossingT[axis] = rayT +
                (voxelToPos(Pos[axis]+1, axis) - gridIntersect[axis]) / ray.d[axis];
            DeltaT[axis] = width[axis] / ray.d[axis];
            Step[axis] = 1;
            Out[axis] = nVoxels[axis];
        }
        else {
            NextCrossingT[axis] = rayT +
                (voxelToPos(Pos[axis], axis) - gridIntersect[axis]) / ray.d[axis];
            DeltaT[axis] = -width[axis] / ray.d[axis];
            Step[axis] = -1;
            Out[axis] = -1;
        }
    }

    pstd::optional<ShapeIntersection> closestHit = {};

    for (;;) {
        for (int idx : voxels[offset(Pos[0], Pos[1], Pos[2])]) {
            pstd::optional<ShapeIntersection> si = primitives[idx].Intersect(ray, tMax);
            if (si && si->tHit < tMax) {
                closestHit = si;
                tMax = si->tHit;
            }
        }

        int bits = ((NextCrossingT[0] < NextCrossingT[1]) << 2) +
                   ((NextCrossingT[0] < NextCrossingT[2]) << 1) +
                   ((NextCrossingT[1] < NextCrossingT[2]));
        const int cmpToAxis[8] = { 2, 1, 2, 1, 2, 2, 0, 0 };
        int stepAxis = cmpToAxis[bits];

        if (tMax < NextCrossingT[stepAxis]) break;
        Pos[stepAxis] += Step[stepAxis];
        if (Pos[stepAxis] == Out[stepAxis]) break;
        NextCrossingT[stepAxis] += DeltaT[stepAxis];
    }

    return closestHit;

}

bool GridAggregate::IntersectP(const Ray &r, Float tMax) const {
    Ray ray = r;
    Float rayT, tMinBounds, tMaxBounds;

    if (!bounds.IntersectP(ray.o, ray.d, tMax, &tMinBounds, &tMaxBounds)){
        return false;
    }
    rayT = tMinBounds;
    Point3f gridIntersect = ray(rayT);

    float NextCrossingT[3], DeltaT[3];
    int Step[3], Out[3], Pos[3];
    for (int axis = 0; axis < 3; ++axis) {
        Pos[axis] = posToVoxel(gridIntersect, axis);
        if (ray.d[axis] == -0.f) ray.d[axis] = 0.f;

        if (ray.d[axis] >= 0) {
            NextCrossingT[axis] = rayT +
                (voxelToPos(Pos[axis]+1, axis) - gridIntersect[axis]) / ray.d[axis];
            DeltaT[axis] = width[axis] / ray.d[axis];
            Step[axis] = 1;
            Out[axis] = nVoxels[axis];
        }
        else {
            NextCrossingT[axis] = rayT +
                (voxelToPos(Pos[axis], axis) - gridIntersect[axis]) / ray.d[axis];
            DeltaT[axis] = -width[axis] / ray.d[axis];
            Step[axis] = -1;
            Out[axis] = -1;
        }
    }

    for (;;) {
        for (int idx : voxels[offset(Pos[0], Pos[1], Pos[2])]) {
            bool si = primitives[idx].IntersectP(ray, tMax);
            if (si) {
                return true;
            }
        }

        int bits = ((NextCrossingT[0] < NextCrossingT[1]) << 2) +
                   ((NextCrossingT[0] < NextCrossingT[2]) << 1) +
                   ((NextCrossingT[1] < NextCrossingT[2]));
        const int cmpToAxis[8] = { 2, 1, 2, 1, 2, 2, 0, 0 };
        int stepAxis = cmpToAxis[bits];

        if (tMax < NextCrossingT[stepAxis]) break;
        Pos[stepAxis] += Step[stepAxis];
        if (Pos[stepAxis] == Out[stepAxis]) break;
        NextCrossingT[stepAxis] += DeltaT[stepAxis];
    }

    return false;
}

GridAggregate* GridAggregate::Create(std::vector<Primitive> prims,
                                        const ParameterDictionary &parameters){
    return new GridAggregate(std::move(prims));
}

}