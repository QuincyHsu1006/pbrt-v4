#ifndef PBRT_CPU_TOON_H
#define PBRT_CPU_TOON_H

#include <pbrt/cpu/integrators.h>

namespace pbrt {

class ToonIntegrator : public RayIntegrator {
  public:
    ToonIntegrator(int level, int maxDepth, Camera camera, Sampler sampler,
                   Primitive aggregate, std::vector<Light> lights);

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda, Sampler sampler,
                       ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const;

    static std::unique_ptr<ToonIntegrator> Create(
        const ParameterDictionary &parameters, Camera camera, Sampler sampler,
        Primitive aggregate, std::vector<Light> lights, const FileLoc *loc);

    std::string ToString() const;

  private:
    Float Quantize(Float value) const;
    SampledSpectrum SampleLd(const SurfaceInteraction &intr, const BSDF *bsdf,
                             SampledWavelengths &lambda, Sampler sampler) const;

    int level;
    int maxDepth;
    UniformLightSampler lightSampler;
};

}  // namespace pbrt

#endif  // PBRT_CPU_TOON_H
