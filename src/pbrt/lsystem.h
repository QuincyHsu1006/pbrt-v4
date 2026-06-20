// pbrt L-system deterministic grammar parser
#ifndef PBRT_LSYSTEM_H
#define PBRT_LSYSTEM_H

#include <pbrt/paramdict.h>
#include <pbrt/pbrt.h>
#include <pbrt/util/vecmath.h>

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace pbrt {

struct BranchSegment {
    Point3f start;
    Point3f end;

    Float radiusStart = 0.05f;
    Float radiusEnd = 0.05f;

    // 0 is the main axis.  Entering '[' increases the active depth by one.
    int depth = 0;

    // Index of the segment from which this segment grows, or -1 for a root.
    int parentSegmentIndex = -1;

    // Position of the F command in the expanded L-system sequence.
    std::size_t sourceSymbolIndex = 0;

    std::string ToString() const;
};

struct LeafInstance {
    Point3f position;

    // Local orientation frame of the leaf after random jitter.
    Vector3f heading;
    Vector3f left;
    Vector3f up;

    Float length = 0.3f;
    Float width = 0.12f;

    int depth = 0;
    std::size_t sourceSymbolIndex = 0;

    std::string ToString() const;
};


// The frame is right-handed:   up = Cross(heading, left)
// Initial frame:               heading = +Y, left = -X, up = +Z
struct TurtleState {
    Point3f position = Point3f(0, 0, 0);

    Vector3f heading = Vector3f(0, 1, 0);
    Vector3f left = Vector3f(-1, 0, 0);
    Vector3f up = Vector3f(0, 0, 1);

    Float branchLength = 1.f;
    Float branchRadius = 0.05f;

    int depth = 0;
    int parentSegmentIndex = -1;

    std::string ToString() const;
};

struct TurtleResult {
    std::vector<BranchSegment> branches;
    std::vector<LeafInstance> leaves;

    TurtleState finalState;
    std::size_t maxStackDepth = 0;

    std::string ToString() const;
};

struct ProductionAlternative {
    Float probability = 1.f;
    std::string successor;
};

struct ProductionSet {
    bool stochastic = false;
    std::vector<ProductionAlternative> alternatives;
};

struct LSystemDefinition {
    // Grammar
    std::string axiom;
    std::unordered_map<char, ProductionSet> productions;

    // Parameters read from the .pbrt file
    int iterations = 0;
    Float angleDegrees = 25.f;
    Float stepLength = 1.f;
    Float radius = 0.05f;
    std::size_t maxSymbols = 1'000'000;

    int seed = 42;
    Float leafLength = 0.18f;
    Float leafWidth = 0.06f;
    Float leafYawJitterDegrees = 10.f;
    Float leafPitchJitterDegrees = 8.f;
    Float leafRollJitterDegrees = 18.f;

    static std::optional<LSystemDefinition> Create(
        const ParameterDictionary &parameters, const FileLoc *loc);

    std::optional<std::string> Expand(const FileLoc *loc) const;

    std::string ToString() const;
};

class TurtleInterpreter {
  public:
    // Supported commands:
    //
    // F  Draw a BranchSegment and move forward.
    // f  Move forward without drawing.
    // +  Yaw left around the turtle's local up axis.
    // -  Yaw right around the turtle's local up axis.
    // &  Pitch down around the turtle's local left axis.
    // ^  Pitch up around the turtle's local left axis.
    // \  Roll left around the turtle's local heading axis.
    // /  Roll right around the turtle's local heading axis.
    // |  Turn around by 180 degrees.
    // [  Push the state, then enter one deeper branch level.
    // ]  Restore the most recently pushed state.
    //
    // Other symbols (for example X or A) are ignored.
    static std::optional<TurtleResult> Interpret(
        const std::string &expanded,
        const LSystemDefinition &definition,
        const FileLoc *loc);
};

}  // namespace pbrt

#endif  // PBRT_LSYSTEM_H
