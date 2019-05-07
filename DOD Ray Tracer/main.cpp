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

struct Ray {
    Vector pos;
    Vector dir;

    Ray() {};
    Ray(const Vector& pos, const Vector& dir) : pos(pos), dir(dir) {};
};

struct RayHit {
    Vector pos;
    Vector norm;
    Vector dir;
    uint8_t hasHit : 1;
};

struct Camera {
    float FOV;
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

void createPrimaryRays(Camera camera, Ray** pRays, int& numRays) {
    int pixelCount = camera.width * camera.height;
    Ray* rays = new Ray[pixelCount];
    *pRays = rays;
    numRays = pixelCount;

    float w_prime = 2 * tanf(degToRad(camera.FOV / 2));
    float h_prime = (w_prime * camera.height) / camera.width;

    float w_start = -w_prime / 2;
    float h_start = h_prime / 2;

    float w_step = w_prime / camera.width;
    float h_step = -h_prime / camera.height;

    for (int i = 0; i < camera.height; ++i) {
        int rowOffset = i * camera.width;
        for (int j = 0; j < camera.width; ++j) {
            int rayIdx = rowOffset + j;
            // NOTE: Primary rays all share the initial camera position.  Maybe primary rays should be
            // a different struct since they'll all share the same position.
            rays[rayIdx].pos = camera.pos;
            rays[rayIdx].dir = Vector(w_start + j * w_step, h_start + i * h_step, -1.f).normalized();
        }
    }
}

bool rayIntersectsSphere(const Ray& r, const Sphere& s, RayHit& h) {
    float a = r.dir.sqmag();
    Vector v = r.pos - s.pos;
    float b = 2 * r.dir.dot(v);
    float c = v.sqmag() - s.rad * s.rad;
    float disc = b * b - 4 * a*c;
    if (disc < 0) return false;
    //we only care about the minus in the plus or minus
    float t = (-b - sqrt(disc)) / (2 * a);
    h.hasHit = false;
    if (t < 0) return false;

    h.pos = r.pos + r.dir * t;
    h.norm = (h.pos - s.pos).normalized();
    h.dir = r.dir;
    h.hasHit = true;
    return true;
}

bool rayIntersectsPlane(Ray r, Plane p, RayHit& h) {
    // Taken from graphicscodex.com
    // t = ((P - C).dot(n)) / (w.dot(n))
    // Ray equation: X(t) = P + t*w
    h.hasHit = false;

    float numerator = (r.pos - p.pos).dot(p.norm);
    float denominator = r.dir.dot(p.norm);
    if (denominator >= 0) return false;
    float t = -numerator / denominator;
    if (t < 0) return false;
    h.norm = p.norm;
    h.pos = r.pos + t * r.dir;
    h.dir = r.dir;
    h.hasHit = true;
    return true;
}

void computeRayHits(Ray* rays, int numRays, Sphere* spheres, int numSpheres, Plane* planes, int numPlanes, RayHit** pRayHits) {
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

void selectReflectableIntersections(RayHit* rayHits, int numRayHits, std::vector<int>** pReflectableRayHits, std::vector<int>* primaryRayIndices, std::vector<int>** pNewPrimaryRayIndices, int& numReflectableRayHits) {
    std::vector<int>* reflectableRayHits = new std::vector<int>();
    *pReflectableRayHits = reflectableRayHits;
    std::vector<int>* newPrimaryRayIndices = new std::vector<int>();
    *pNewPrimaryRayIndices = newPrimaryRayIndices;

    reflectableRayHits->reserve(numRayHits);
    newPrimaryRayIndices->reserve(numRayHits);

    int numReflectable = 0;

    for (int i = 0; i < numRayHits; ++i) {
        if (rayHits[i].hasHit) {
            reflectableRayHits->push_back(i);
            newPrimaryRayIndices->push_back((*primaryRayIndices)[i]);
            ++numReflectable;
        }
    }

    numReflectableRayHits = numReflectable;
    reflectableRayHits->shrink_to_fit();
    newPrimaryRayIndices->shrink_to_fit();
}

void computeReflectionRays(RayHit* rayHits, int numRayHits, Ray** pReflectionRays) {
    Ray* reflectionRays = new Ray[numRayHits];
    *pReflectionRays = reflectionRays;

    for (int i = 0; i < numRayHits; ++i) {
        RayHit hit = rayHits[i];

        Vector v = -hit.dir.normalized();
        Vector n = hit.norm.normalized();
        Vector direction = ((n * (2 * v.dot(n))) - v).normalized();
        reflectionRays[i] = Ray(hit.pos + 0.001f * direction, direction);
    }
}

void computePointLightDiffuse(RayHit* rayHits, int numRayHits, PointLight* lights, int numLights, Vector* diffuseColor, Sphere* spheres, int numSpheres, Plane* planes, int numPlanes) {
    for (int rayHitIdx = 0; rayHitIdx < numRayHits; ++rayHitIdx) {
        RayHit hit = rayHits[rayHitIdx];

        if (!hit.hasHit) {
            continue;
        }

        for (int lightIdx = 0; lightIdx < numLights; ++lightIdx) {
            PointLight light = lights[lightIdx];
            Vector lightDiff = light.pos - hit.pos;
            float lightDistanceSquared = lightDiff.sqmag();
            Vector lightDir = lightDiff.normalized();

            Ray shadowRay(hit.pos + hit.norm * 0.001f, lightDir);
            RayHit shadowHit;
            bool hasHit = false;

            // Spheres
            for (int sphereIdx = 0; sphereIdx < numSpheres && !hasHit; ++sphereIdx) {
                if (rayIntersectsSphere(shadowRay, spheres[sphereIdx], shadowHit)) {
                    if ((shadowHit.pos - hit.pos).sqmag() < lightDistanceSquared) {
                        hasHit = true;
                    }
                }
            }

            // Planes
            for (int planeIdx = 0; planeIdx < numPlanes && !hasHit; ++planeIdx) {
                if (rayIntersectsPlane(shadowRay, planes[planeIdx], shadowHit)) {
                    if ((shadowHit.pos - hit.pos).sqmag() < lightDistanceSquared) {
                        hasHit = true;
                    }
                }
            }

            if (!hasHit) {
                diffuseColor[rayHitIdx] = diffuseColor[rayHitIdx] + light.color * max(hit.norm.dot(lightDir), 0.f);
            }
        }
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
    Camera camera;
    camera.FOV = 50.f;
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