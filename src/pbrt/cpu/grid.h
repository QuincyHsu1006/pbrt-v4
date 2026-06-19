#ifndef PBRT_CPU_GRID_H
#define PBRT_CPU_GRID_H

#include <pbrt/pbrt.h>

#include <pbrt/cpu/primitive.h>
#include <pbrt/util/parallel.h>

#include <atomic>
#include <memory>
#include <vector>

namespace pbrt{

class GridAggregate {
    public:
        GridAggregate(std::vector<Primitive> p);

        static GridAggregate *Create(std::vector<Primitive> prims,
                                const ParameterDictionary &parameters);
        // ~GridAggregate();

        Bounds3f Bounds() const { return bounds; }
        pstd::optional<ShapeIntersection> Intersect(const Ray &ray, Float tMax) const;
        bool IntersectP(const Ray &ray, Float tMax) const;

    private:
        int posToVoxel(const Point3f &P, int axis) const {
            int v = (int)((P[axis] - bounds.pMin[axis]) *
                            invWidth[axis]);
            return Clamp(v, 0, nVoxels[axis]-1);
        }
        float voxelToPos(int p, int axis) const {
            return bounds.pMin[axis] + p * width[axis];
        }
        inline int offset(int x, int y, int z) const {
            return z*nVoxels[0]*nVoxels[1] + y*nVoxels[0] + x;
        }

        // GridAccel Private Data
        std::vector<Primitive> primitives;
        int nVoxels[3];
        Bounds3f bounds;
        Vector3f width, invWidth;
        // Voxel **voxels;
        std::vector<std::vector<int>> voxels;
};

}

#endif