#include <pbrt/cpu/toon.h>

#include <pbrt/bsdf.h>
#include <pbrt/cameras.h>
#include <pbrt/interaction.h>
#include <pbrt/lights.h>
#include <pbrt/paramdict.h>
#include <pbrt/util/error.h>
#include <pbrt/util/math.h>
#include <pbrt/util/print.h>

#include <cmath>

namespace pbrt {

ToonIntegrator::ToonIntegrator(int level, int maxDepth, Camera camera, Sampler sampler,
                               Primitive aggregate, std::vector<Light> lights)
    : RayIntegrator(camera, sampler, aggregate, lights),
      level(level),
      maxDepth(maxDepth),
      lightSampler(lights, Allocator()) {}

Float ToonIntegrator::Quantize(Float value) const {
    value = Clamp(value, Float(0), Float(1));
    if (level == 1)
        return value > 0 ? 1 : 0;
    return std::floor(value * (level - 1) + Float(0.5)) / (level - 1);
}

SampledSpectrum ToonIntegrator::Li(RayDifferential ray, SampledWavelengths &lambda,
                                   Sampler sampler, ScratchBuffer &scratchBuffer,
                                   VisibleSurface *visibleSurface) const {
    SampledSpectrum L(0.f), beta(1.f);
    int depth = 0;

    Float p_b = 0, etaScale = 1;
    bool specularBounce = false;
    LightSampleContext prevIntrCtx;

    while (true) {
        pstd::optional<ShapeIntersection> si = Intersect(ray);
        if (!si) {
            for (const auto &light : infiniteLights) {
                SampledSpectrum Le = light.Le(ray, lambda);
                if (depth == 0 || specularBounce)
                    L += beta * Le;
                else {
                    Float p_l = lightSampler.PMF(prevIntrCtx, light) *
                                light.PDF_Li(prevIntrCtx, ray.d, true);
                    Float w_b = PowerHeuristic(1, p_b, 1, p_l);
                    L += beta * w_b * Le;
                }
            }
            break;
        }

        SampledSpectrum Le = si->intr.Le(-ray.d, lambda);
        if (Le) {
            if (depth == 0 || specularBounce)
                L += beta * Le;
            else {
                Light areaLight(si->intr.areaLight);
                Float p_l = lightSampler.PMF(prevIntrCtx, areaLight) *
                            areaLight.PDF_Li(prevIntrCtx, ray.d, true);
                Float w_l = PowerHeuristic(1, p_b, 1, p_l);
                L += beta * w_l * Le;
            }
        }

        SurfaceInteraction &isect = si->intr;
        BSDF bsdf = isect.GetBSDF(ray, lambda, camera, scratchBuffer, sampler);
        if (!bsdf) {
            specularBounce = true;
            isect.SkipIntersection(&ray, si->tHit);
            continue;
        }

        if (depth == 0 && visibleSurface) {
            constexpr int nRhoSamples = 4;
            const Float ucRho[nRhoSamples] = {0.25, 0.5, 0.75, 0.875};
            const Point2f uRho[nRhoSamples] = {
                Point2f(0.25, 0.25), Point2f(0.75, 0.25), Point2f(0.25, 0.75),
                Point2f(0.75, 0.75)};
            SampledSpectrum albedo = bsdf.rho(isect.wo, ucRho, uRho);
            *visibleSurface = VisibleSurface(isect, albedo, lambda);
        }

        if (depth++ == maxDepth)
            break;

        if (IsNonSpecular(bsdf.Flags()))
            L += beta * SampleLd(isect, &bsdf, lambda, sampler);

        Vector3f wo = -ray.d;
        pstd::optional<BSDFSample> bs = bsdf.Sample_f(wo, sampler.Get1D(), sampler.Get2D());
        if (!bs)
            break;

        beta *= bs->f * Quantize(AbsDot(bs->wi, isect.shading.n)) / bs->pdf;
        p_b = bs->pdfIsProportional ? bsdf.PDF(wo, bs->wi) : bs->pdf;
        specularBounce = bs->IsSpecular();
        if (bs->IsTransmission())
            etaScale *= Sqr(bs->eta);

        prevIntrCtx = si->intr;
        ray = isect.SpawnRay(ray, bsdf, bs->wi, bs->flags, bs->eta);

        SampledSpectrum rrBeta = beta * etaScale;
        if (rrBeta.MaxComponentValue() < 1 && depth > 1) {
            Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
            if (sampler.Get1D() < q)
                break;
            beta /= 1 - q;
        }
    }

    return L;
}

SampledSpectrum ToonIntegrator::SampleLd(const SurfaceInteraction &intr, const BSDF *bsdf,
                                         SampledWavelengths &lambda,
                                         Sampler sampler) const {
    LightSampleContext ctx(intr);
    BxDFFlags flags = bsdf->Flags();
    if (IsReflective(flags) && !IsTransmissive(flags))
        ctx.pi = intr.OffsetRayOrigin(intr.wo);
    else if (IsTransmissive(flags) && !IsReflective(flags))
        ctx.pi = intr.OffsetRayOrigin(-intr.wo);

    pstd::optional<SampledLight> sampledLight =
        lightSampler.Sample(ctx, sampler.Get1D());
    if (!sampledLight)
        return {};

    Light light = sampledLight->light;
    pstd::optional<LightLiSample> ls =
        light.SampleLi(ctx, sampler.Get2D(), lambda, true);
    if (!ls || !ls->L || ls->pdf == 0)
        return {};

    Vector3f wi = ls->wi;
    Float nDotL = Quantize(AbsDot(wi, intr.shading.n));
    SampledSpectrum f = bsdf->f(intr.wo, wi) * nDotL;
    if (nDotL == 0 || !f || !Unoccluded(intr, ls->pLight))
        return {};

    Float p_l = sampledLight->p * ls->pdf;
    if (IsDeltaLight(light.Type()))
        return ls->L * f / p_l;

    Float p_b = bsdf->PDF(intr.wo, wi);
    Float w_l = PowerHeuristic(1, p_l, 1, p_b);
    return w_l * ls->L * f / p_l;
}

std::unique_ptr<ToonIntegrator> ToonIntegrator::Create(
    const ParameterDictionary &parameters, Camera camera, Sampler sampler,
    Primitive aggregate, std::vector<Light> lights, const FileLoc *loc) {
    int level = parameters.GetOneInt("level", 3);
    int maxDepth = parameters.GetOneInt("maxdepth", 5);
    if (level < 1)
        ErrorExit(loc, "\"level\" must be at least 1 for the \"toon\" integrator.");
    if (maxDepth < 0)
        ErrorExit(loc, "\"maxdepth\" must be non-negative for the \"toon\" integrator.");
    return std::make_unique<ToonIntegrator>(level, maxDepth, camera, sampler, aggregate,
                                            lights);
}

std::string ToonIntegrator::ToString() const {
    return StringPrintf("[ ToonIntegrator level: %d maxDepth: %d ]", level, maxDepth);
}

}  // namespace pbrt
