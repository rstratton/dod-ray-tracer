#include "pch.h"
#define _USE_MATH_DEFINES
#include "math.h"
#include <iostream>
#include <vector>
#include <limits>
#include <immintrin.h>
#include <intrin.h>
#include <assert.h>

#pragma warning(disable:4996)

inline float max(float a, float b) {
    return a > b ? a : b;
}

inline float min(float a, float b) {
    return a < b ? a : b;
}

static inline __m256i unpack_24b_shufmask(unsigned int packed_mask) {
    // alternative: broadcast / variable-shift.  Requires a vector constant, though.
    // This strategy is probably better if the packed mask is coming directly from memory, esp. on Skylake where variable-shift is cheap
    __m256i indices = _mm256_set1_epi32(packed_mask);
    __m256i m = _mm256_sllv_epi32(indices, _mm256_setr_epi32(29, 26, 23, 20, 17, 14, 11, 8));
    __m256i shufmask = _mm256_srli_epi32(m, 29);
    return shufmask;
}

__m256 pack_ps(__m256 src, uint8_t mask) {
    // Taken from https://stackoverflow.com/questions/36932240/avx2-what-is-the-most-efficient-way-to-pack-left-based-on-a-mask
    unsigned expanded_mask = _pdep_u32(mask, 0b001'001'001'001'001'001'001'001);
    expanded_mask += (expanded_mask<<1) + (expanded_mask<<2);  // ABC -> AAABBBCCC: triplicate each bit
    // + instead of | lets the compiler implement it with a multiply by 7

    const unsigned identity_indices = 0b111'110'101'100'011'010'001'000;    // the identity shuffle for vpermps, packed
    unsigned wanted_indices = _pext_u32(identity_indices, expanded_mask);   // just the indices we want, contiguous in the low 24 bits or less.

    // unpack the same as we would for the LUT version
    __m256i shufmask = unpack_24b_shufmask(wanted_indices);
    return _mm256_permutevar8x32_ps(src, shufmask);
}

__m256i pack_epi32(__m256i src, uint8_t mask) {
    // Adapted from https://stackoverflow.com/questions/36932240/avx2-what-is-the-most-efficient-way-to-pack-left-based-on-a-mask
    unsigned expanded_mask = _pdep_u32(mask, 0b001'001'001'001'001'001'001'001);
    expanded_mask += (expanded_mask<<1) + (expanded_mask<<2);  // ABC -> AAABBBCCC: triplicate each bit
    // + instead of | lets the compiler implement it with a multiply by 7

    const unsigned identity_indices = 0b111'110'101'100'011'010'001'000;    // the identity shuffle for vpermps, packed
    unsigned wanted_indices = _pext_u32(identity_indices, expanded_mask);   // just the indices we want, contiguous in the low 24 bits or less.

    // unpack the same as we would for the LUT version
    __m256i shufmask = unpack_24b_shufmask(wanted_indices);
    return _mm256_permutevar8x32_epi32(src, shufmask);
}

struct Vector {
    float x;
    float y;
    float z;

    Vector() {};
    Vector(float x, float y, float z) : x(x), y(y), z(z) {};
    Vector(const Vector& v) : x(v.x), y(v.y), z(v.z) {};

    Vector operator+(const Vector& o) const {
        return Vector(
            x + o.x,
            y + o.y,
            z + o.z
        );
    }

    Vector& operator+=(const Vector& o) {
        x += o.x;
        y += o.y;
        z += o.z;
        return *this;
    }

    Vector operator-(const Vector& o) const {
        return Vector(
            x - o.x,
            y - o.y,
            z - o.z
        );
    }

    Vector operator-() const {
        return Vector(-x, -y, -z);
    }

    Vector operator*(float f) const {
        return Vector(
            f * x,
            f * y,
            f * z
        );
    }

    Vector operator/(float f) const {
        return Vector(
            x / f,
            y / f,
            z / f
        );
    }

    Vector normalized() const {
        return (*this) / mag();
    }

    float mag() const {
        return sqrt(sqmag());
    }

    float sqmag() const {
        return x * x + y * y + z * z;
    }

    float dot(const Vector& o) const {
        return x * o.x + y * o.y + z * o.z;
    }
};

Vector operator*(float f, const Vector& v) {
    return Vector(
        f * v.x,
        f * v.y,
        f * v.z
    );
}

struct VectorAVX {
    __m256 x;
    __m256 y;
    __m256 z;

    VectorAVX() {};

    VectorAVX(const Vector& v) {
        x = _mm256_set1_ps(v.x);
        y = _mm256_set1_ps(v.y);
        z = _mm256_set1_ps(v.z);
    }

    VectorAVX(__m256 x, __m256 y, __m256 z) : x(x), y(y), z(z) {}

    VectorAVX operator+(const VectorAVX& o) const {
        return VectorAVX(
            _mm256_add_ps(x, o.x),
            _mm256_add_ps(y, o.y),
            _mm256_add_ps(z, o.z)
        );
    }

    VectorAVX operator-(const VectorAVX& o) const {
        return VectorAVX(
            _mm256_sub_ps(x, o.x),
            _mm256_sub_ps(y, o.y),
            _mm256_sub_ps(z, o.z)
        );
    }

    VectorAVX operator-() const {
        __m256 negOne = _mm256_set1_ps(-1.f);
        return VectorAVX(
            _mm256_mul_ps(x, negOne),
            _mm256_mul_ps(y, negOne),
            _mm256_mul_ps(z, negOne)
        );
    }

    VectorAVX operator*(float f) const {
        __m256 multiplier = _mm256_set1_ps(f);
        return VectorAVX(
            _mm256_mul_ps(x, multiplier),
            _mm256_mul_ps(y, multiplier),
            _mm256_mul_ps(z, multiplier)
        );
    }

    VectorAVX operator*(__m256 f) const {
        return VectorAVX(
            _mm256_mul_ps(x, f),
            _mm256_mul_ps(y, f),
            _mm256_mul_ps(z, f)
        );
    }

    VectorAVX operator/(float f) const {
        __m256 divisor = _mm256_set1_ps(f);
        return VectorAVX(
            _mm256_div_ps(x, divisor),
            _mm256_div_ps(y, divisor),
            _mm256_div_ps(z, divisor)
        );
    }

    VectorAVX normalized() const {
        __m256 divisor = _mm256_sqrt_ps(sqmag());
        return VectorAVX(
            _mm256_div_ps(x, divisor),
            _mm256_div_ps(y, divisor),
            _mm256_div_ps(z, divisor)
        );
    }

    __m256 mag() const {
        return _mm256_sqrt_ps(sqmag());
    }

    __m256 sqmag() const {
        __m256 xx = _mm256_mul_ps(x, x);
        __m256 yy = _mm256_mul_ps(y, y);
        __m256 zz = _mm256_mul_ps(z, z);
        __m256 xx_yy = _mm256_add_ps(xx, yy);
        return _mm256_add_ps(xx_yy, zz);
    }

    __m256 dot(const VectorAVX& o) const {
        __m256 xx = _mm256_mul_ps(x, o.x);
        __m256 yy = _mm256_mul_ps(y, o.y);
        __m256 zz = _mm256_mul_ps(z, o.z);
        __m256 xx_yy = _mm256_add_ps(xx, yy);
        return _mm256_add_ps(xx_yy, zz);
    }

    void blend(VectorAVX& o, __m256 mask) {
        x = _mm256_blendv_ps(x, o.x, mask);
        y = _mm256_blendv_ps(y, o.y, mask);
        z = _mm256_blendv_ps(z, o.z, mask);
    }

    VectorAVX pack(uint8_t mask) {
        return VectorAVX(
            pack_ps(x, mask),
            pack_ps(y, mask),
            pack_ps(z, mask)
        );
    }
};

struct RayAVX {
    VectorAVX pos;
    VectorAVX dir;
    __m256i primaryRayIdx;

    RayAVX() {};
    RayAVX(const VectorAVX& pos, const VectorAVX& dir, __m256i idx) : pos(pos), dir(dir), primaryRayIdx(idx) {};
};

struct RayHitAVX {
    VectorAVX pos;
    VectorAVX norm;
    VectorAVX dir;
    __m256i primaryRayIdx;

    RayHitAVX() {};
    RayHitAVX(const VectorAVX& pos, const VectorAVX& norm, const VectorAVX& dir, __m256i primaryRayIdx) : pos(pos), norm(norm), dir(dir), primaryRayIdx(primaryRayIdx) {};

    void blend(RayHitAVX o, __m256 mask) {
        pos.blend(o.pos, mask);
        norm.blend(o.norm, mask);
        dir.blend(o.dir, mask);
        // No need to blend primaryRayIdx.  `blend` is used to take one of two intersections
        // based on which is closer, which doesn't change which primary ray the ray
        // hit is correlated with.
    }

    RayHitAVX pack(uint8_t hitFlags) {
        RayHitAVX result;
        result.pos = pos.pack(hitFlags);
        result.dir = dir.pack(hitFlags);
        result.norm = norm.pack(hitFlags);
        result.primaryRayIdx = pack_epi32(primaryRayIdx, hitFlags);
        return result;
    }
};

struct VectorSOA {
    float* x;
    float* y;
    float* z;
    size_t size;

    VectorSOA() = delete;
    VectorSOA(const VectorSOA& other) = delete;
    void operator=(const VectorSOA& other) = delete;
    VectorSOA(size_t size) : size(size) {
        // Allocate in multiples of 8 so that we can use AVX
        size_t malloc_size = ((int)ceil(sizeof(float) * size / 8.f)) * 8;
        x = (float*)_aligned_malloc(malloc_size, 32);
        y = (float*)_aligned_malloc(malloc_size, 32);
        z = (float*)_aligned_malloc(malloc_size, 32);
    }
    ~VectorSOA() {
        _aligned_free(x);
        _aligned_free(y);
        _aligned_free(z);
    }

    void storeAligned(const VectorAVX& vec, size_t index) {
        assert(index % 8 == 0);
        _mm256_store_ps(x + index, vec.x);
        _mm256_store_ps(y + index, vec.y);
        _mm256_store_ps(z + index, vec.z);
    }

    VectorAVX loadAligned(size_t index) {
        assert(index % 8 == 0);
        return VectorAVX(
            _mm256_load_ps(x + index),
            _mm256_load_ps(y + index),
            _mm256_load_ps(z + index)
        );
    }

    void storeUnaligned(const VectorAVX& vec, size_t index) {
        _mm256_storeu_ps(x + index, vec.x);
        _mm256_storeu_ps(y + index, vec.y);
        _mm256_storeu_ps(z + index, vec.z);
    }

    VectorAVX gather(__m256i indices) {
        return VectorAVX(
            _mm256_i32gather_ps(x, indices, 1),
            _mm256_i32gather_ps(y, indices, 1),
            _mm256_i32gather_ps(z, indices, 1)
        );
    }
};

struct RaySOA {
    VectorSOA pos;
    VectorSOA dir;
    int* primaryRayIdx; // Array mapping RaySOA indices to primary-ray/pixel indices
    size_t size;

    RaySOA() = delete;
    RaySOA(const RaySOA& other) = delete;
    void operator=(const RaySOA& other) = delete;
    RaySOA(size_t size) : size(size), pos(size), dir(size) {
        size_t malloc_size = ((int)ceil(sizeof(int) * size / 8.f)) * 8;
        primaryRayIdx = (int*)_aligned_malloc(malloc_size, 32);
    };
    ~RaySOA() {
        _aligned_free(primaryRayIdx);
    }

    void storeAligned(const RayAVX& ray, size_t index) {
        pos.storeAligned(ray.pos, index);
        dir.storeAligned(ray.dir, index);
        _mm256_store_si256((__m256i*)(primaryRayIdx + index), ray.primaryRayIdx);
    }

    RayAVX loadAligned(size_t index) {
        assert(index % 8 == 0);
        return RayAVX(
            pos.loadAligned(index),
            dir.loadAligned(index),
            _mm256_load_si256((__m256i*)(primaryRayIdx + index))
        );
    }
};

struct RayHitSOA {
    VectorSOA pos; // Location where the ray intersected the object
    VectorSOA dir; // Direction of the incident ray
    VectorSOA norm; // Normal of the surface
    int* primaryRayIdx; // Array mapping RayHitSOA indices to primary-ray/pixel indices
    size_t size;

    RayHitSOA() = delete;
    RayHitSOA(const RayHitSOA& other) = delete;
    void operator=(const RayHitSOA& other) = delete;
    RayHitSOA(size_t size) : size(size), pos(size), dir(size), norm(size) {
        size_t malloc_size = ((int)ceil(sizeof(int) * size / 8.f)) * 8;
        primaryRayIdx = (int*)_aligned_malloc(malloc_size, 32);
    };
    ~RayHitSOA() {
        _aligned_free(primaryRayIdx);
    }

    void storeAligned(const RayHitAVX& rayHit, size_t index) {
        assert(index % 8 == 0);
        pos.storeAligned(rayHit.pos, index);
        dir.storeAligned(rayHit.dir, index);
        norm.storeAligned(rayHit.norm, index);
        _mm256_store_si256((__m256i*)(primaryRayIdx + index), rayHit.primaryRayIdx);
    }

    void storeUnaligned(const RayHitAVX& rayHit, size_t index) {
        pos.storeUnaligned(rayHit.pos, index);
        dir.storeUnaligned(rayHit.dir, index);
        norm.storeUnaligned(rayHit.norm, index);
        _mm256_storeu_si256((__m256i*)(primaryRayIdx + index), rayHit.primaryRayIdx);
    }

    RayHitAVX loadAligned(size_t index) {
        assert(index % 8 == 0);
        return RayHitAVX(
            pos.loadAligned(index),
            norm.loadAligned(index),
            dir.loadAligned(index),
            _mm256_load_si256((__m256i*)(primaryRayIdx + index))
        );
    }
};

struct Camera {
    float verticalFov;
    int width;
    int height;
    Vector pos;
};

struct Plane {
    Vector pos;
    Vector norm;
};

struct Sphere {
    Vector pos;
    float rad;
};

struct Color {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

struct PointLight {
    Vector pos;
    Vector color;
};


float degToRad(float deg) {
    return deg * M_PI / 180.f;
}

void createPrimaryRays(Camera camera, RaySOA** pRays, int& numRays) {
    numRays = camera.width * camera.height;
    RaySOA* rays = new RaySOA(numRays);
    *pRays = rays;

    float verticalImagePlaneSize = 2 * tanf(degToRad(camera.verticalFov / 2));
    float horizontalImagePlaneSize = (verticalImagePlaneSize / camera.height) * camera.width;

    __m256 x_0 = _mm256_set1_ps(-horizontalImagePlaneSize / 2);
    __m256 y_0 = _mm256_set1_ps(verticalImagePlaneSize / 2);

    __m256 dx = _mm256_set1_ps(horizontalImagePlaneSize / camera.width);
    __m256 dy = _mm256_set1_ps(-verticalImagePlaneSize / camera.height);

    VectorAVX cameraPos(_mm256_set1_ps(camera.pos.x), _mm256_set1_ps(camera.pos.y), _mm256_set1_ps(camera.pos.z));
    __m256 negOne = _mm256_set1_ps(-1.f);
    __m256i idxOffset = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    int limit = (int)ceil(numRays / 8.f) * 8;
    for (int i = 0; i < limit; i += 8) {
        __m256i rayIdx = _mm256_add_epi32(_mm256_set1_epi32(i), idxOffset);

        // TODO: is there some more clever way to do this in AVX using floating point ops and rounding?
        __m256 xIndex = _mm256_set_ps(
            (i + 0) % camera.width,
            (i + 1) % camera.width,
            (i + 2) % camera.width,
            (i + 3) % camera.width,
            (i + 4) % camera.width,
            (i + 5) % camera.width,
            (i + 6) % camera.width,
            (i + 7) % camera.width
        );

        __m256 yIndex = _mm256_set_ps(
            (i + 0) / camera.width,
            (i + 1) / camera.width,
            (i + 2) / camera.width,
            (i + 3) / camera.width,
            (i + 4) / camera.width,
            (i + 5) / camera.width,
            (i + 6) / camera.width,
            (i + 7) / camera.width
        );

        __m256 xOffset = _mm256_mul_ps(xIndex, dx);
        __m256 yOffset = _mm256_mul_ps(yIndex, dy);
        __m256 x = _mm256_add_ps(x_0, xOffset);
        __m256 y = _mm256_add_ps(y_0, yOffset);

        RayAVX ray(cameraPos, VectorAVX(x, y, negOne).normalized(), rayIdx);
        rays->storeAligned(ray, i);
    }
}

__m256 rayIntersectsSphereAVX(const RayAVX& ray, const Sphere& sphere, RayHitAVX& rayHit) {
    VectorAVX spherePos = VectorAVX(sphere.pos);
    __m256 sphereRad = _mm256_set1_ps(sphere.rad);
    // Vector v = r.pos - s.pos;
    VectorAVX v = ray.pos - spherePos;
    // float b = 2 * r.dir.dot(v);
    __m256 b = _mm256_mul_ps(_mm256_set1_ps(2.f), ray.dir.dot(v));
    // float c = v.sqmag() - s.rad * s.rad;
    __m256 c = _mm256_sub_ps(v.sqmag(), _mm256_mul_ps(sphereRad, sphereRad));
    // float disc = b * b - 4 * a*c;
    __m256 disc = _mm256_sub_ps(_mm256_mul_ps(b, b), _mm256_mul_ps(_mm256_set1_ps(4.f), c));
    // if (disc < 0) return false;
    // Bits are 1 if disc >= 0, and 0 otherwise.
    __m256 discMask = _mm256_cmp_ps(_mm256_setzero_ps(), disc, _CMP_LT_OS);

    uint8_t discFlags = _mm256_movemask_ps(discMask);

    // To prevent taking the sqrt of a negative number, load 0 in place of disc[i], if disc[i] < 0.
    // We will ignore this value later anyway, since disc < 0 means we don't have an intersection.
    __m256 clampedDisc = _mm256_blendv_ps(_mm256_setzero_ps(), disc, discMask);

    // float t = (-b - sqrt(disc)) / (2 * a);
    __m256 sqrtDisc = _mm256_sqrt_ps(clampedDisc);
    __m256 negBMinusSqrtDisc = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(-1.f), b), sqrtDisc);
    // Divide by 2*a (which is just 2, because a is 1)
    __m256 t = _mm256_mul_ps(negBMinusSqrtDisc, _mm256_set1_ps(0.5f));
    // if (t < 0) return false;
    // Bits are 1 if t >= 0, and 0 otherwise
    __m256 tMask = _mm256_cmp_ps(_mm256_setzero_ps(), t, _CMP_LT_OS);

    // Even though some of these rays didn't actually have a hit, populate all hits anyway
    // to avoid branching.
    VectorAVX hitPos = ray.pos + ray.dir * t;
    VectorAVX hitNorm = (hitPos - spherePos).normalized();
    rayHit.pos = hitPos;
    rayHit.norm = hitNorm;
    rayHit.dir = ray.dir;

    // Bits are 1 if we found an intersection, 0 otherwise
    return _mm256_and_ps(discMask, tMask);
}

__m256 rayIntersectsPlaneAVX(const RayAVX& ray, const Plane& plane, RayHitAVX& rayHit) {
    VectorAVX planePos(
        _mm256_set1_ps(plane.pos.x),
        _mm256_set1_ps(plane.pos.y),
        _mm256_set1_ps(plane.pos.z)
    );

    VectorAVX planeNorm(
        _mm256_set1_ps(plane.norm.x),
        _mm256_set1_ps(plane.norm.y),
        _mm256_set1_ps(plane.norm.z)
    );

    // float numerator = (r.pos - p.pos).dot(p.norm);
    __m256 numerator = _mm256_mul_ps(_mm256_set1_ps(-1.f), (ray.pos - planePos).dot(planeNorm));
    // float denominator = r.dir.dot(p.norm);
    __m256 denominator = ray.dir.dot(planeNorm);
    // if (denominator >= 0) return false;
    // Bits are 1 if denominator < 0, and 0 otherwise
    __m256 denomLessThanZeroMask = _mm256_cmp_ps(denominator, _mm256_setzero_ps(), _CMP_LT_OS);

    // To prevent possible division by 0, load 1 in place of denominator[i] if denominator[i] >= 0.
    // This value will be ignored later anyway, since denominator >= 0 means we don't have an
    // intersection.
    __m256 maskedDenom = _mm256_blendv_ps(_mm256_set1_ps(1.f), denominator, denomLessThanZeroMask);
    // float t = -numerator / denominator;
    __m256 t = _mm256_div_ps(numerator, maskedDenom);
    // if (t < 0) return false;
    // Bits are 1 if t >= 0, and 0 otherwise
    __m256 tMask = _mm256_cmp_ps(_mm256_setzero_ps(), t, _CMP_LT_OS);

    // Even though some of these rays didn't actually have a hit, populate all hits anyway
    // to avoid branching.
    VectorAVX hitPos = ray.pos + ray.dir * t;
    rayHit.pos = hitPos;
    rayHit.norm = planeNorm;
    rayHit.dir = ray.dir;

    // Bits are 1 if we found an intersection, 0 otherwise
    return _mm256_and_ps(denomLessThanZeroMask, tMask);
}

void computeRayIntersections(RaySOA* rays, int numRays, Sphere* spheres, int numSpheres, Plane* planes, int numPlanes, RayHitSOA** pRayHits, int& numIntersections) {
    numIntersections = 0;
    RayHitSOA* rayHits = new RayHitSOA(numRays);
    *pRayHits = rayHits;
    
    const __m256 infinity = _mm256_set1_ps(std::numeric_limits<float>::infinity());

    int limit = (int)ceil(numRays / 8.f) * 8;
    for (int i = 0; i < limit; i += 8) {
        RayAVX ray(rays->loadAligned(i));
        RayHitAVX newHit;
        RayHitAVX closestHit;
        closestHit.primaryRayIdx = ray.primaryRayIdx;
        uint8_t hitFlags = 0; // 1 if hit, 0 if no hit
        __m256 closestHitDistanceSquared = infinity;

        // Find closest intersections

        // Spheres
        for (int sphereIdx = 0; sphereIdx < numSpheres; ++sphereIdx) {
            __m256 newHitMask = rayIntersectsSphereAVX(ray, spheres[sphereIdx], newHit);
            __m256 newHitDistanceSquared = _mm256_blendv_ps(infinity, (newHit.pos - ray.pos).sqmag(), newHitMask);
            __m256 newHitDistanceLessThanClosest = _mm256_cmp_ps(newHitDistanceSquared, closestHitDistanceSquared, _CMP_LT_OS);
            closestHitDistanceSquared = _mm256_blendv_ps(closestHitDistanceSquared, newHitDistanceSquared, newHitDistanceLessThanClosest);
            closestHit.blend(newHit, newHitDistanceLessThanClosest);
            uint8_t newHitFlags = _mm256_movemask_ps(newHitMask);
            hitFlags = hitFlags | newHitFlags;
        }

        // Planes
        for (int planeIdx = 0; planeIdx < numPlanes; ++planeIdx) {
            __m256 newHitMask = rayIntersectsPlaneAVX(ray, planes[planeIdx], newHit);
            __m256 newHitDistanceSquared = _mm256_blendv_ps(infinity, (newHit.pos - ray.pos).sqmag(), newHitMask);
            __m256 newHitDistanceLessThanClosest = _mm256_cmp_ps(newHitDistanceSquared, closestHitDistanceSquared, _CMP_LT_OS);
            closestHitDistanceSquared = _mm256_blendv_ps(closestHitDistanceSquared, newHitDistanceSquared, newHitDistanceLessThanClosest);
            closestHit.blend(newHit, newHitDistanceLessThanClosest);
            uint8_t newHitFlags = _mm256_movemask_ps(newHitMask);
            hitFlags = hitFlags | newHitFlags;
        }

        // Pack and store intersections
        RayHitAVX packedHit(closestHit.pack(hitFlags));
        rayHits->storeUnaligned(packedHit, numIntersections);
        numIntersections += __popcnt16(hitFlags);
    }
}

void computeMirrorRays(RayHitSOA* hits, int numIntersections, RaySOA** pMirrorRays) {
    RaySOA* mirrorRays = new RaySOA(numIntersections);
    *pMirrorRays = mirrorRays;

    int limit = (int)ceil(numIntersections / 8.f) * 8;
    for (int i = 0; i < limit; i += 8) {
        RayHitAVX hit(hits->loadAligned(i));
        VectorAVX v = hit.dir.normalized() * -1.f;
        VectorAVX n = hit.norm.normalized();
        VectorAVX direction = ((n * v.dot(n) * 2.f) - v).normalized();
        RayAVX ray(hit.pos + direction * 0.001f, direction, hit.primaryRayIdx);
        mirrorRays->storeAligned(ray, i);
    }
}

void computeDirectIllumination(VectorSOA** pRadiance, RayHitSOA* hits, int numIntersections, PointLight* lights, int numLights, Sphere* spheres, int numSpheres, Plane* planes, int numPlanes) {
    VectorSOA* radiance = new VectorSOA(numIntersections);
    for (int i = 0; i < numIntersections; ++i) {
        radiance->x[i] = 0.f;
        radiance->y[i] = 0.f;
        radiance->z[i] = 0.f;
    }
    *pRadiance = radiance;
    
    const __m256 infinity = _mm256_set1_ps(std::numeric_limits<float>::infinity());

    int limit = (int)ceil(numIntersections / 8.f) * 8;
    for (int i = 0; i < limit; i += 8) {
        RayHitAVX hit(hits->loadAligned(i));

        for (int lightIdx = 0; lightIdx < numLights; ++lightIdx) {
            PointLight light = lights[lightIdx];
            __m256i zero(_mm256_setzero_si256());

            VectorAVX lightPos(
                _mm256_set1_ps(light.pos.x),
                _mm256_set1_ps(light.pos.y),
                _mm256_set1_ps(light.pos.z)
            );

            VectorAVX lightDiff = lightPos - hit.pos;
            __m256 lightDistanceSquared = lightDiff.sqmag();
            VectorAVX lightDir = lightDiff.normalized();

            // Test if light is occluded
            RayAVX shadowRay(hit.pos + (hit.norm * 0.001f), lightDir, zero);
            RayHitAVX shadowHit;
            __m256 isShadowed = _mm256_setzero_ps();
            uint8_t shadowFlags = 0;


            // Spheres
            for (int sphereIdx = 0; sphereIdx < numSpheres && shadowFlags != 0xFF; ++sphereIdx) {
                __m256 shadowHitMask = rayIntersectsSphereAVX(shadowRay, spheres[sphereIdx], shadowHit);
                __m256 shadowHitDistanceSquared = _mm256_blendv_ps(infinity, (shadowHit.pos - hit.pos).sqmag(), shadowHitMask);
                __m256 shadowHitDistanceLessThanLight = _mm256_cmp_ps(shadowHitDistanceSquared, lightDistanceSquared, _CMP_LT_OS);
                isShadowed = _mm256_or_ps(isShadowed, shadowHitDistanceLessThanLight);
                shadowFlags = _mm256_movemask_ps(isShadowed);
            }

            // Planes
            for (int planeIdx = 0; planeIdx < numPlanes && shadowFlags != 0xFF; ++planeIdx) {
                __m256 shadowHitMask = rayIntersectsPlaneAVX(shadowRay, planes[planeIdx], shadowHit);
                __m256 shadowHitDistanceSquared = _mm256_blendv_ps(infinity, (shadowHit.pos - hit.pos).sqmag(), shadowHitMask);
                __m256 shadowHitDistanceLessThanLight = _mm256_cmp_ps(shadowHitDistanceSquared, lightDistanceSquared, _CMP_LT_OS);
                isShadowed = _mm256_or_ps(isShadowed, shadowHitDistanceLessThanLight);
                shadowFlags = _mm256_movemask_ps(isShadowed);
            }

            VectorAVX lightColor(
                _mm256_set1_ps(light.color.x),
                _mm256_set1_ps(light.color.y),
                _mm256_set1_ps(light.color.z)
            );

            VectorAVX currentRadiance(radiance->loadAligned(i));

            __m256 lightMultiplier = _mm256_max_ps(hit.norm.dot(lightDir), _mm256_setzero_ps());
            VectorAVX diffuseRadiance = lightColor * lightMultiplier;
            VectorAVX maskedDiffuse(
                _mm256_blendv_ps(diffuseRadiance.x, _mm256_setzero_ps(), isShadowed),
                _mm256_blendv_ps(diffuseRadiance.y, _mm256_setzero_ps(), isShadowed),
                _mm256_blendv_ps(diffuseRadiance.z, _mm256_setzero_ps(), isShadowed)
            );

            radiance->storeAligned(currentRadiance + maskedDiffuse, i);
        }
    }
}

//void hitsToRadiance(Vector* radiance, int numPixels, RayHitSOA* hits, int numIntersections) {
//    const Vector white(1.f, 1.f, 1.f);
//    const Vector black(0.f, 0.f, 0.f);
//
//    for (int i = 0; i < numPixels; ++i) {
//        radiance[i] = black;
//    }
//
//    for (int i = 0; i < numIntersections; ++i) {
//        int rayIndex = hits->primaryRayIdx[i];
//        radiance[rayIndex] = white;
//    }
//}

void accumulateRadiance(VectorSOA* accumulator, VectorSOA* newValues, int numNewValues, int* indexMap) {
    for (int i = 0; i < numNewValues; ++i) {
        accumulator->x[indexMap[i]] += newValues->x[i];
        accumulator->y[indexMap[i]] += newValues->x[i];
        accumulator->z[indexMap[i]] += newValues->x[i];
    }
}

void convertRadianceToPixels(VectorSOA* radiance, unsigned char **pPixels, int numPixels) {
    unsigned char *pixels = new unsigned char[3 * numPixels];
    *pPixels = pixels;

    for (int i = 0; i < numPixels; ++i) {
        Vector value = Vector(radiance->x[i], radiance->y[i], radiance->z[i]);
        pixels[3 * i + 0] = (int)min(value.x * 255, 255);
        pixels[3 * i + 1] = (int)min(value.y * 255, 255);
        pixels[3 * i + 2] = (int)min(value.z * 255, 255);
    }
}

void writePPM(unsigned char *buf, int width, int height, const char *fn) {
    FILE *fp = fopen(fn, "wb");
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    for (int i = 0; i < width*height * 3; ++i) {
        fputc(buf[i], fp);
    }
    fclose(fp);
    printf("Wrote image file %s\n", fn);
}

int main()
{
    // Init scene
    Camera camera;
    camera.verticalFov = 50.f;
    camera.width = 1000;
    camera.height = 1000;
    camera.pos = { 0.f, 6.f, 20.f };

    int numSpheresHorizontal = 5;
    int numSpheresVertical = 5;
    int numSpheres = numSpheresHorizontal * numSpheresVertical;
    Sphere* spheres = new Sphere[numSpheres];

    for (int i = 0; i < numSpheresHorizontal; ++i) {
        for (int j = 0; j < numSpheresVertical; ++j) {
            Sphere s;
            s.pos = Vector((i * 3) - 6, 0, -j * 3);
            s.rad = 1.f;
            spheres[i * numSpheresHorizontal + j] = s;
        }
    }

    int numPlanes = 6;
    Plane* planes = new Plane[numPlanes];
    planes[0] = { { 0.f, -1.f, 0.f }, { 0.f, 1.f, 0.f } };
    planes[1] = { { 0.f, 0.f, -30.f }, { 0.f, 0.f, 1.f } };
    planes[2] = { { -20.f, 0.f, 0.f }, { 1.f, 0.f, 0.f } };
    planes[3] = { { 20.f, 0.f, 0.f }, { -1.f, 0.f, 0.f } };
    planes[4] = { { 0.f, 40.f, 0.f }, { 0.f, -1.f, 0.f } };
    planes[5] = { { 0.f, 0.f, 39.f }, { 0.f, 0.f, -1.f } };

    int numLights = 2;
    PointLight* pointLights = new PointLight[numLights];
    pointLights[0] = { { 19.f, 19.f, 1.f }, { 0.07f, 0.07f, 0.05f } };
    pointLights[1] = { { -19.f, 4.f, 4.f }, { 0.05f, 0.05f, 0.07f } };
    
    // Compute image
    VectorSOA* radiance = new VectorSOA(camera.width * camera.height);
    for (int i = 0; i < camera.width * camera.height; ++i) {
        radiance->x[i] = 0.f;
        radiance->y[i] = 0.f;
        radiance->z[i] = 0.f;
    }

    // Create primary rays
    RaySOA* primaryRays;
    int numRays;
    createPrimaryRays(camera, &primaryRays, numRays);

    // Compute primary ray hits
    RayHitSOA* rayHits;
    int numIntersections;
    computeRayIntersections(primaryRays, numRays, spheres, numSpheres, planes, numPlanes, &rayHits, numIntersections);

    delete primaryRays;

    // Compute direct illumination for primary ray intersections
    VectorSOA* directIllumination;
    computeDirectIllumination(&directIllumination, rayHits, numIntersections, pointLights, numLights, spheres, numSpheres, planes, numPlanes);

    // Incorporate direct illumination term into total radiance
    accumulateRadiance(radiance, directIllumination, numIntersections, rayHits->primaryRayIdx);

    // Compute radiance from reflections
    for (int reflectionNum = 0; reflectionNum < 10; ++reflectionNum) {
        // Compute mirror rays from previous batch of intersections
        RaySOA* mirrorRays;
        int numRays = numIntersections;
        computeMirrorRays(rayHits, numRays, &mirrorRays);

        // Compute mirror ray intersections
        RayHitSOA* mirrorRayHits;
        computeRayIntersections(mirrorRays, numRays, spheres, numSpheres, planes, numPlanes, &mirrorRayHits, numIntersections);

        // Compute direct illumination
        VectorSOA* directIllumination;
        computeDirectIllumination(&directIllumination, mirrorRayHits, numIntersections, pointLights, numLights, spheres, numSpheres, planes, numPlanes);

        // Incorporate direct illumination term into total radiance
        accumulateRadiance(radiance, directIllumination, numIntersections, mirrorRayHits->primaryRayIdx);

        delete rayHits;
        rayHits = mirrorRayHits;
    }

    unsigned char* pixels;
    convertRadianceToPixels(radiance, &pixels, numRays);
    writePPM(pixels, camera.width, camera.height, "..\\renders\\image.ppm");

    return 0;
}