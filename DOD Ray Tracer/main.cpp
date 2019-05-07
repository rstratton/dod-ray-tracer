#include "pch.h"
#define _USE_MATH_DEFINES
#include "math.h"
#include <iostream>
#include <vector>
#include <limits>

#pragma warning(disable:4996)

inline float max(float a, float b) {
    return a > b ? a : b;
}

inline float min(float a, float b) {
    return a < b ? a : b;
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

struct VectorAVX {
    float x[8];
    float y[8];
    float z[8];
};

struct RayAVX {
    VectorAVX pos;
    VectorAVX dir;
};

struct RayHitAVX {
    VectorAVX pos;
    VectorAVX norm;
    VectorAVX dir;
    bool hasHit[8];
};

float degToRad(float deg) {
    return deg * M_PI / 180.f;
}

void createPrimaryRaysAVX(Camera camera, RayAVX** pRays, int& numRays) {
    int pixelCount = camera.width * camera.height;
    int rayCount = ceil(pixelCount / 8.f);
    RayAVX* rays = new RayAVX[rayCount];
    *pRays = rays;
    numRays = rayCount;

    float verticalImagePlaneSize = 2 * tanf(degToRad(camera.verticalFov / 2));
    float horizontalImagePlaneSize = (verticalImagePlaneSize / camera.height) * camera.width;

    float x_0 = -horizontalImagePlaneSize / 2;
    float y_0 = verticalImagePlaneSize / 2;

    float dx = horizontalImagePlaneSize / camera.width;
    float dy = -verticalImagePlaneSize / camera.height;

    for (int i = 0; i < pixelCount; i += 8) {
        int rayIdx = i / 8;

        for (int j = 0; j < 8; ++j) {
            float x = x_0 + ((i + j) % camera.width) * dx;
            float y = y_0 + ((i + j) / camera.width) * dy;

            Vector v = Vector(x, y, -1.f).normalized();

            rays[rayIdx].dir.x[j] = v.x;
            rays[rayIdx].dir.y[j] = v.y;
            rays[rayIdx].dir.z[j] = v.z;

            rays[rayIdx].pos.x[j] = camera.pos.x;
            rays[rayIdx].pos.y[j] = camera.pos.y;
            rays[rayIdx].pos.z[j] = camera.pos.z;
        }
    }
}

void computeRayHits(RayAVX* rays, int numRays, Sphere* spheres, int numSpheres, Plane* planes, int numPlanes, RayHitAVX** pRayHits) {
    // We will have at most `numRays` hits
    RayHit* rayHits = new RayHit[numRays];
    *pRayHits = rayHits;

    for (int rayIdx = 0; rayIdx < numRays; ++rayIdx) {
        Ray ray = rays[rayIdx];
        RayHit newHit, closestHit;
        float closestHitDistanceSquared = std::numeric_limits<float>::infinity();

        // Spheres
        for (int sphereIdx = 0; sphereIdx < numSpheres; ++sphereIdx) {
            if (rayIntersectsSphere(ray, spheres[sphereIdx], newHit)) {
                float newHitDistanceSquared = (newHit.pos - ray.pos).sqmag();
                if (closestHitDistanceSquared > newHitDistanceSquared) {
                    closestHit = newHit;
                    closestHitDistanceSquared = newHitDistanceSquared;
                }
            }
        }

        // Planes
        for (int planeIdx = 0; planeIdx < numPlanes; ++planeIdx) {
            if (rayIntersectsPlane(ray, planes[planeIdx], newHit)) {
                float newHitDistanceSquared = (newHit.pos - ray.pos).sqmag();

                if (closestHitDistanceSquared > newHitDistanceSquared) {
                    closestHit = newHit;
                    closestHitDistanceSquared = newHitDistanceSquared;
                }
            }
        }

        rayHits[rayIdx] = closestHit;
    }
}


void convertDiffuseToPixels(Vector* diffuse, unsigned char **pPixels, int numPixels) {
    unsigned char *pixels = new unsigned char[3 * numPixels];
    *pPixels = pixels;

    for (int i = 0; i < numPixels; ++i) {
        Vector value = diffuse[i];
        pixels[3 * i]     = (int) min(value.x * 255, 255);
        pixels[3 * i + 1] = (int) min(value.y * 255, 255);
        pixels[3 * i + 2] = (int) min(value.z * 255, 255);
    }
}

void integrateReflection(Vector* diffuse, Vector* refDiffuse, int numReflectionRays, std::vector<int>& primaryRayIndices) {
    for (int i = 0; i < numReflectionRays; ++i) {
        int diffIdx = primaryRayIndices[i];
        diffuse[diffIdx] = diffuse[diffIdx] + refDiffuse[i];
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
    
    // Create primary rays
    int numRays;
    Ray* rays;
    createPrimaryRays(camera, &rays, numRays);

    // Compute primary ray hits
    RayHit* rayHits;
    computeRayHits(rays, numRays, spheres, numSpheres, planes, numPlanes, &rayHits);

    delete[] rays;

    // Compute direct illumination for primary ray hits
    Vector* diffuse = new Vector[numRays];
    for (int i = 0; i < numRays; ++i) {
        diffuse[i] = { 0.f, 0.f, 0.f };
    }
    computePointLightDiffuse(rayHits, numRays, pointLights, numLights, diffuse, spheres, numSpheres, planes, numPlanes);

    // Initialize map from rayHit index to primaryRay/pixel index (initialized to identity map)
    std::vector<int>* primaryRayIndices = new std::vector<int>();
    primaryRayIndices->resize(numRays);
    for (int i = 0; i < numRays; ++i) {
        (*primaryRayIndices)[i] = i;
    }

    int curNumRays = numRays;

    // Compute contribution from reflections
    for (int i = 0; i < 10; ++i) {
        // Select rays which intersected with scene objects
        std::vector<int>* reflectableRayHitIndices;
        std::vector<int>* newPrimaryRayIndices;
        int numReflectableRayHits;
        selectReflectableIntersections(rayHits, curNumRays, &reflectableRayHitIndices, primaryRayIndices, &newPrimaryRayIndices, numReflectableRayHits);
        delete primaryRayIndices;
        primaryRayIndices = newPrimaryRayIndices;

        RayHit* reflectableRayHits = new RayHit[numReflectableRayHits];
        for (int j = 0; j < numReflectableRayHits; ++j) {
            reflectableRayHits[j] = rayHits[(*reflectableRayHitIndices)[j]];
        }

        delete[] rayHits;

        Ray* reflectionRays;
        computeReflectionRays(reflectableRayHits, numReflectableRayHits, &reflectionRays);

        delete[] reflectableRayHits;

        // Compute reflection ray hits
        RayHit* reflectionRayHits;
        computeRayHits(reflectionRays, numReflectableRayHits, spheres, numSpheres, planes, numPlanes, &reflectionRayHits);

        delete[] reflectionRays;

        // Compute diffuse color for reflection ray hits
        Vector* refDiffuse = new Vector[numReflectableRayHits];
        for (int i = 0; i < numReflectableRayHits; ++i) {
            refDiffuse[i] = { 0.f, 0.f, 0.f };
        }
        computePointLightDiffuse(reflectionRayHits, numReflectableRayHits, pointLights, numLights, refDiffuse, spheres, numSpheres, planes, numPlanes);

        rayHits = reflectionRayHits;
        curNumRays = numReflectableRayHits;

        // Integrate diffuse from reflection rays with primary rays
        integrateReflection(diffuse, refDiffuse, numReflectableRayHits, *primaryRayIndices);
        delete[] refDiffuse;
    }

    unsigned char* pixels;
    convertDiffuseToPixels(diffuse, &pixels, numRays);
    writePPM(pixels, camera.width, camera.height, "..\\renders\\image.ppm");

    return 0;
}