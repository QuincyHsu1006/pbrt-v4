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

struct Module {
    char symbol = '\0';
    std::vector<Float> parameters;

    std::string ToString() const;
};

using ParametricWord = std::vector<Module>;

std::string ParametricWordToString(const ParametricWord &word);

struct ExpressionToken {
    enum class Type {
        Number,
        Variable,
        Add,
        Subtract,
        Multiply,
        Divide,
        Power,
        Negate
    };

    Type type = Type::Number;
    Float number = 0.f;
    std::string variable;
};

struct Expression {
    std::vector<ExpressionToken> tokens;

    std::optional<Float> Evaluate(
        const std::unordered_map<std::string, Float> &environment,
        const FileLoc *loc) const;

    std::string ToString() const;
};

struct ModulePattern {
    char symbol = '\0';
    std::vector<std::string> formalParameters;

    std::string ToString() const;
};

struct ModuleTemplate {
    char symbol = '\0';
    std::vector<Expression> parameterExpressions;

    std::optional<Module> Instantiate(
        const std::unordered_map<std::string, Float> &environment,
        const FileLoc *loc) const;

    std::string ToString() const;
};

struct ProductionAlternative {
    Float probability = 1.f;
    ModulePattern predecessor;
    std::vector<ModuleTemplate> successor;
};

struct ProductionSet {
    char symbol = '\0';
    std::size_t arity = 0;
    bool stochastic = false;
    std::vector<ProductionAlternative> alternatives;
};

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
    std::size_t sourceModuleIndex = 0;

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

    TurtleState finalState;
    std::size_t maxStackDepth = 0;

    std::string ToString() const;
};



struct LSystemDefinition {
    // Grammar
    ParametricWord axiom;
    std::unordered_map<char, std::vector<ProductionSet>> productions;

    std::unordered_map<char, std::size_t> variableArities;
    std::unordered_map<std::string, Float> constants;

    // Parameters read from the .pbrt file
    int iterations = 0;
    Float angleDegrees = 25.f;
    Float stepLength = 1.f;
    Float radius = 0.05f;
    std::size_t maxModules = 1'000'000;

    Vector3f verticalDirection = Vector3f(0, 1, 0);
    Vector3f tropismVector = Vector3f(0, 0, 0);
    Float tropismSusceptibility = 0.f;

    int seed = 42;
    int grammarSeed = 42;

    static std::optional<LSystemDefinition> Create(
        const ParameterDictionary &parameters, const FileLoc *loc);

    std::optional<ParametricWord> Expand(const FileLoc *loc) const;

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
        const ParametricWord &expanded,
        const LSystemDefinition &definition,
        const FileLoc *loc);
};

}  // namespace pbrt

#endif  // PBRT_LSYSTEM_H
