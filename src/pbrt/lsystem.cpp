// pbrt L-system deterministic grammar parser
#include <pbrt/lsystem.h>

#include <pbrt/util/error.h>
#include <pbrt/util/math.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace pbrt {
namespace {

constexpr Float FrameEpsilon = 1e-12f;
constexpr Float ProbabilityNormalizationTolerance = 1e-4f;

std::string Trim(std::string s) {
    auto isNotSpace = [](unsigned char c) { return !std::isspace(c); };

    auto begin = std::find_if(s.begin(), s.end(), isNotSpace);
    auto end = std::find_if(s.rbegin(), s.rend(), isNotSpace).base();

    if (begin >= end)
        return {};
    return std::string(begin, end);
}

std::string RemoveWhitespace(const std::string &s) {
    std::string result;
    result.reserve(s.size());

    for (unsigned char c : s) {
        if (!std::isspace(c))
            result.push_back(static_cast<char>(c));
    }
    return result;
}

std::string ToLower(std::string s) {
    for (char &c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool IsValidPredecessor(char c) {
    return std::isalpha(static_cast<unsigned char>(c)) != 0;
}

bool ValidateBrackets(const std::string &sequence, const FileLoc *loc) {
    int depth = 0;

    for (std::size_t i = 0; i < sequence.size(); ++i) {
        if (sequence[i] == '[') {
            ++depth;
        } else if (sequence[i] == ']') {
            --depth;
            if (depth < 0) {
                Error(loc,
                      "L-system expansion has an unmatched ']' at symbol %d.",
                      int(i));
                return false;
            }
        }
    }

    if (depth != 0) {
        Error(loc,
              "L-system expansion has %d unmatched '[' bracket(s).",
              depth);
        return false;
    }

    return true;
}

bool ParseProbability(const std::string &text, Float *probability) {
    if (text.empty())
        return false;

    errno = 0;
    char *end = nullptr;
    double value = std::strtod(text.c_str(), &end);

    if (end == text.c_str() ||
        end == nullptr ||
        *end != '\0' ||
        errno == ERANGE ||
        !std::isfinite(value)) {
        return false;
    }

    *probability = Float(value);
    return true;
}

Float Uniform01(std::mt19937 *rng) {
    constexpr double InvTwoTo32 = 1.0 / 4294967296.0;
    return Float(double((*rng)()) * InvTwoTo32);
}

const ProductionAlternative &ChooseAlternative(
    const ProductionSet &set, std::mt19937 *rng) {
    if (!set.stochastic)
        return set.alternatives.front();

    Float u = Uniform01(rng);
    Float cumulative = 0.f;

    for (const ProductionAlternative &alternative : set.alternatives) {
        cumulative += alternative.probability;
        if (u < cumulative)
            return alternative;
    }

    // The probability sum is validated during parsing. This fallback handles
    // the final few floating-point ulps near one.
    return set.alternatives.back();
}

Float DegreesToRadians(Float degrees) {
    return degrees * Pi / 180.f;
}

// Rodrigues' rotation formula.  The axis is expected to be nonzero.
Vector3f RotateAroundAxis(Vector3f vector, Vector3f axis,
                          Float radians) {
    axis = Normalize(axis);

    Float cosTheta = std::cos(radians);
    Float sinTheta = std::sin(radians);

    return cosTheta * vector +
           sinTheta * Cross(axis, vector) +
           (1 - cosTheta) * Dot(axis, vector) * axis;
}

bool IsFinite(Point3f p) {
    return std::isfinite(p.x) &&
           std::isfinite(p.y) &&
           std::isfinite(p.z);
}

bool IsFinite(Vector3f v) {
    return std::isfinite(v.x) &&
           std::isfinite(v.y) &&
           std::isfinite(v.z);
}

// Remove accumulated floating-point drift from the local turtle frame.
bool Reorthonormalize(TurtleState *state, const FileLoc *loc,
                      std::size_t symbolIndex) {
    if (!IsFinite(state->heading) || !IsFinite(state->left) || !IsFinite(state->up) ||
        LengthSquared(state->heading) <= FrameEpsilon) {
        Error(loc, "Invalid turtle frame after symbol %d.", int(symbolIndex));
        return false;
    }

    state->heading = Normalize(state->heading);

    // Gram-Schmidt: remove the heading component from left.
    state->left =
        state->left - Dot(state->left, state->heading) * state->heading;

    if (!IsFinite(state->left) || LengthSquared(state->left) <= FrameEpsilon) {
        Error(loc, "Degenerate turtle left vector after symbol %d.", int(symbolIndex));
        return false;
    }

    state->left = Normalize(state->left);
    state->up = Cross(state->heading, state->left);

    if (!IsFinite(state->up) || LengthSquared(state->up) <= FrameEpsilon) {
        Error(loc, "Degenerate turtle up vector after symbol %d.", int(symbolIndex));
        return false;
    }

    state->up = Normalize(state->up);

    // Recompute left one more time so all three vectors are mutually
    // perpendicular and preserve the intended handedness.
    state->left = Normalize(Cross(state->up, state->heading));

    return true;
}

bool Yaw(TurtleState *state, Float radians,
         const FileLoc *loc, std::size_t symbolIndex) {
    state->heading = RotateAroundAxis(state->heading, state->up, radians);
    state->left = RotateAroundAxis(state->left, state->up, radians);
    return Reorthonormalize(state, loc, symbolIndex);
}

bool Pitch(TurtleState *state, Float radians,
           const FileLoc *loc, std::size_t symbolIndex) {
    state->heading = RotateAroundAxis(state->heading, state->left, radians);
    state->up = RotateAroundAxis(state->up, state->left, radians);
    return Reorthonormalize(state, loc, symbolIndex);
}

bool Roll(TurtleState *state, Float radians,
          const FileLoc *loc, std::size_t symbolIndex) {
    state->left = RotateAroundAxis(state->left, state->heading, radians);
    state->up = RotateAroundAxis(state->up, state->heading, radians);
    return Reorthonormalize(state, loc, symbolIndex);
}

bool ValidateNonnegativeAngle(Float degrees, const char *name,
                              const FileLoc *loc) {
    if (!std::isfinite(degrees) || degrees < 0.f || degrees > 180.f) {
        Error(loc, "L-system \"%s\" must be finite and in [0, 180]; got %f.", name, degrees);
        return false;
    }
    return true;
}

}  // namespace

std::string BranchSegment::ToString() const {
    std::ostringstream out;
    out << "[ BranchSegment"
        << " start: " << start.ToString()
        << " end: " << end.ToString()
        << " radiusStart: " << radiusStart
        << " radiusEnd: " << radiusEnd
        << " depth: " << depth
        << " parentSegmentIndex: " << parentSegmentIndex
        << " sourceSymbolIndex: " << sourceSymbolIndex
        << " ]";
    return out.str();
}

std::string LeafInstance::ToString() const {
    std::ostringstream out;
    out << "[ LeafInstance"
        << " position: " << position.ToString()
        << " heading: " << heading.ToString()
        << " left: " << left.ToString()
        << " up: " << up.ToString()
        << " length: " << length
        << " width: " << width
        << " depth: " << depth
        << " sourceSymbolIndex: " << sourceSymbolIndex
        << " ]";
    return out.str();
}

std::string TurtleState::ToString() const {
    std::ostringstream out;
    out << "[ TurtleState"
        << " position: " << position.ToString()
        << " heading: " << heading.ToString()
        << " left: " << left.ToString()
        << " up: " << up.ToString()
        << " branchLength: " << branchLength
        << " branchRadius: " << branchRadius
        << " depth: " << depth
        << " parentSegmentIndex: " << parentSegmentIndex
        << " ]";
    return out.str();
}

std::string TurtleResult::ToString() const {
    std::ostringstream out;
    out << "[ TurtleResult"
        << " branchCount: " << branches.size()
        << " maxStackDepth: " << maxStackDepth
        << " finalState: " << finalState.ToString()
        << " ]";
    return out.str();
}


std::optional<LSystemDefinition> LSystemDefinition::Create(
    const ParameterDictionary &parameters, const FileLoc *loc) {
    LSystemDefinition definition;

    // Expected syntax:
    //
    // "string grammar" [
    //     "axiom: F"
    //     "F -> F[+F]F[-F]F"
    // ]
    std::vector<std::string> grammarLines =
        parameters.GetStringArray("grammar");

    definition.iterations = parameters.GetOneInt("iterations", 0);
    definition.angleDegrees = parameters.GetOneFloat("angle", 25.f);
    definition.stepLength = parameters.GetOneFloat("length", 1.f);
    definition.radius = parameters.GetOneFloat("radius", 0.05f);
    definition.seed = parameters.GetOneInt("seed", 1);

    definition.leafLength = parameters.GetOneFloat("leaflength", 0.18f);
    definition.leafWidth = parameters.GetOneFloat("leafwidth", 0.06f);
    definition.leafYawJitterDegrees =
        parameters.GetOneFloat("leafyawjitter", 10.f);
    definition.leafPitchJitterDegrees =
        parameters.GetOneFloat("leafpitchjitter", 8.f);
    definition.leafRollJitterDegrees =
        parameters.GetOneFloat("leafrolljitter", 18.f);

    int maxSymbols = parameters.GetOneInt("maxsymbols", 1'000'000);

    if (grammarLines.empty()) {
        Error(loc,
              "Shape \"lsystem\" requires a non-empty "
              "\"string grammar\" parameter.");
        return std::nullopt;
    }

    if (definition.iterations < 0 || definition.iterations > 64) {
        Error(loc,
              "L-system \"iterations\" must be between 0 and 64; got %d.",
              definition.iterations);
        return std::nullopt;
    }

    if (!std::isfinite(definition.angleDegrees) ||
        definition.angleDegrees < 0.f ||
        definition.angleDegrees > 360.f) {
        Error(loc,
              "L-system \"angle\" must be finite and in [0, 360]; got %f.",
              definition.angleDegrees);
        return std::nullopt;
    }

    if (!std::isfinite(definition.stepLength) || definition.stepLength <= 0.f) {
        Error(loc,
              "L-system \"length\" must be finite and greater than zero; "
              "got %f.",
              definition.stepLength);
        return std::nullopt;
    }

    if (!std::isfinite(definition.radius) || definition.radius <= 0.f) {
        Error(loc,
              "L-system \"radius\" must be finite and greater than zero; "
              "got %f.",
              definition.radius);
        return std::nullopt;
    }

    if (!std::isfinite(definition.leafLength) || definition.leafLength <= 0.f) {
        Error(loc,
              "L-system \"leaflength\" must be finite and greater than zero; "
              "got %f.",
              definition.leafLength);
        return std::nullopt;
    }

    if (!std::isfinite(definition.leafWidth) || definition.leafWidth <= 0.f) {
        Error(loc,
              "L-system \"leafwidth\" must be finite and greater than zero; "
              "got %f.",
              definition.leafWidth);
        return std::nullopt;
    }

    if (!ValidateNonnegativeAngle(definition.leafYawJitterDegrees,
                                  "leafyawjitter", loc) ||
        !ValidateNonnegativeAngle(definition.leafPitchJitterDegrees,
                                  "leafpitchjitter", loc) ||
        !ValidateNonnegativeAngle(definition.leafRollJitterDegrees,
                                  "leafrolljitter", loc)) {
        return std::nullopt;
    }

    if (maxSymbols <= 0) {
        Error(loc,
              "L-system \"maxsymbols\" must be greater than zero; got %d.",
              maxSymbols);
        return std::nullopt;
    }

    definition.maxSymbols = static_cast<std::size_t>(maxSymbols);

    bool foundAxiom = false;

    for (std::size_t lineIndex = 0; lineIndex < grammarLines.size(); ++lineIndex) {
        std::string line = grammarLines[lineIndex];

        // Allow comments inside grammar strings.
        if (std::size_t comment = line.find('#');
            comment != std::string::npos) {
            line.erase(comment);
        }

        line = Trim(std::move(line));
        if (line.empty())
            continue;

        // Parse "axiom: ..." or "axiom = ...".
        std::size_t colon = line.find(':');
        std::size_t equal = line.find('=');
        std::size_t axiomSeparator = std::string::npos;

        if (colon != std::string::npos)
            axiomSeparator = colon;
        if (equal != std::string::npos &&
            (axiomSeparator == std::string::npos || equal < axiomSeparator))
            axiomSeparator = equal;

        if (axiomSeparator != std::string::npos) {
            std::string key = ToLower(Trim(line.substr(0, axiomSeparator)));

            if (key == "axiom") {
                if (foundAxiom) {
                    Error(loc,
                          "L-system grammar contains more than one axiom "
                          "declaration.");
                    return std::nullopt;
                }

                definition.axiom =
                    RemoveWhitespace(line.substr(axiomSeparator + 1));

                if (definition.axiom.empty()) {
                    Error(loc, "L-system axiom may not be empty.");
                    return std::nullopt;
                }

                foundAxiom = true;
                continue;
            }
        }

        // Parse a deterministic production: "F -> successor".
        std::size_t arrow = line.find("->");
        if (arrow == std::string::npos) {
            Error(loc,
                  "Invalid L-system grammar entry %d: \"%s\". Expected "
                  "\"axiom: ...\" or \"A -> ...\".",
                  int(lineIndex + 1), line.c_str());
            return std::nullopt;
        }

        std::string lhs = RemoveWhitespace(line.substr(0, arrow));
        std::string rhs = RemoveWhitespace(line.substr(arrow + 2));

        std::size_t probabilitySeparator = lhs.find(':');
        bool hasExplicitProbability = (probabilitySeparator != std::string::npos);

        std::string predecessorText =
            hasExplicitProbability
                ? lhs.substr(0, probabilitySeparator) : lhs;

        if (predecessorText.size() != 1 ||
            !IsValidPredecessor(predecessorText[0])) {
            Error(loc,
                  "Invalid predecessor \"%s\" in L-system grammar entry %d. "
                  "This version requires one alphabetic symbol.",
                  predecessorText.c_str(), int(lineIndex + 1));
            return std::nullopt;
        }

        if (hasExplicitProbability &&
            lhs.find(':', probabilitySeparator + 1) !=
                std::string::npos) {
            Error(loc,
                  "Invalid probability syntax in grammar entry %d: \"%s\".",
                  int(lineIndex + 1), line.c_str());
            return std::nullopt;
        }

        Float probability = 1.f;

        if (hasExplicitProbability) {
            std::string probabilityText =
                lhs.substr(probabilitySeparator + 1);

            if (!ParseProbability(probabilityText, &probability) ||
                probability <= 0.f) {
                Error(loc,
                      "Invalid stochastic weight \"%s\" in grammar "
                      "entry %d. Expected a finite value greater than zero.",
                      probabilityText.c_str(),
                      int(lineIndex + 1));
                return std::nullopt;
            }
        }

        if (!ValidateBrackets(rhs, loc)) {
            return std::nullopt;
        }

        char predecessor = predecessorText[0];

        auto [productionIter, inserted] =
            definition.productions.try_emplace(predecessor);

        ProductionSet &set = productionIter->second;

        if (inserted) {
            set.stochastic = hasExplicitProbability;
        } else if (set.stochastic != hasExplicitProbability) {
            Error(loc,
                  "Cannot mix deterministic and stochastic productions for "
                  "symbol '%c'. Either use one \"A -> ...\" rule or use "
                  "\"A:probability -> ...\" for every alternative.",
                  predecessor);
            return std::nullopt;
        }

        if (!set.stochastic && !set.alternatives.empty()) {
            Error(loc,
                  "Duplicate deterministic production for symbol '%c'.",
                  predecessor);
            return std::nullopt;
        }

        set.alternatives.push_back(
            ProductionAlternative{
                probability,
                std::move(rhs)
            });
    }

    if (!foundAxiom) {
        Error(loc,
              "L-system grammar must contain an axiom declaration such as "
              "\"axiom: F\".");
        return std::nullopt;
    }

    for (auto &[predecessor, set] : definition.productions) {
        if (set.alternatives.empty()) {
            Error(loc,
                  "L-system production set for '%c' is empty.",
                  predecessor);
            return std::nullopt;
        }

        if (!set.stochastic) {
            if (set.alternatives.size() != 1 ||
                set.alternatives[0].probability != 1.f) {
                Error(loc,
                      "Invalid deterministic production set for '%c'.",
                      predecessor);
                return std::nullopt;
            }

            continue;
        }

        double weightSum = 0.0;
        for (const ProductionAlternative &alternative :
             set.alternatives) {
            weightSum += double(alternative.probability);
        }

        if (!std::isfinite(weightSum) || weightSum <= 0.0) {
            Error(loc,
                  "Stochastic weights for symbol '%c' have invalid "
                  "sum %f.",
                  predecessor,
                  weightSum);
            return std::nullopt;
        }

        if (std::abs(weightSum - 1.0) >
            double(ProbabilityNormalizationTolerance)) {
            Warning(loc,
                    "Stochastic weights for symbol '%c' sum to %f; "
                    "normalizing them to one.",
                    predecessor,
                    weightSum);
        }

        Float inverseWeightSum = Float(1.0 / weightSum);
        for (ProductionAlternative &alternative :
             set.alternatives) {
            alternative.probability *= inverseWeightSum;
        }
    }

    if (definition.axiom.size() >
        definition.maxSymbols) {
        Error(loc,
              "L-system axiom contains %d symbols, exceeding "
              "maxsymbols=%d.",
              int(definition.axiom.size()),
              int(definition.maxSymbols));
        return std::nullopt;
    }

    return definition;
}

std::optional<std::string> LSystemDefinition::Expand(
    const FileLoc *loc) const {
    std::string current = axiom;

    std::mt19937 grammarRng(
        static_cast<std::mt19937::result_type>(
            seed));

    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::string next;

        // Most grammars grow rather than shrink. Reserving the current size
        // avoids repeated allocations without making an unsafe growth guess.
        next.reserve(current.size());

        for (char symbol : current) {
            auto productionIter =
                productions.find(symbol);

            if (productionIter == productions.end()) {
                if (next.size() == maxSymbols) {
                    Error(loc,
                          "L-system expansion exceeds maxsymbols=%d while "
                          "computing iteration %d.",
                          int(maxSymbols),
                          iteration + 1);
                    return std::nullopt;
                }

                next.push_back(symbol);
                continue;
            }

            const ProductionAlternative &alternative =
                ChooseAlternative(
                    productionIter->second,
                    &grammarRng);

            if (alternative.successor.size() >
                maxSymbols - next.size()) {
                Error(loc,
                      "L-system expansion exceeds maxsymbols=%d while "
                      "computing iteration %d.",
                      int(maxSymbols),
                      iteration + 1);
                return std::nullopt;
            }

            next.append(alternative.successor);
        }

        current = std::move(next);
    }

    if (!ValidateBrackets(current, loc))
        return std::nullopt;

    return current;
}

std::string LSystemDefinition::ToString() const {
    std::vector<char> predecessors;
    predecessors.reserve(productions.size());

    for (const auto &[predecessor, unused] : productions)
        predecessors.push_back(predecessor);

    std::sort(predecessors.begin(),
              predecessors.end());

    std::ostringstream out;
    out << "[ LSystemDefinition"
        << " axiom: \"" << axiom << "\""
        << " iterations: " << iterations
        << " angleDegrees: " << angleDegrees
        << " stepLength: " << stepLength
        << " radius: " << radius
        << " seed: " << seed
        << " leafLength: " << leafLength
        << " leafWidth: " << leafWidth
        << " leafYawJitterDegrees: "
        << leafYawJitterDegrees
        << " leafPitchJitterDegrees: "
        << leafPitchJitterDegrees
        << " leafRollJitterDegrees: "
        << leafRollJitterDegrees
        << " maxSymbols: " << maxSymbols
        << " productions: {";

    bool firstSet = true;

    for (char predecessor : predecessors) {
        if (!firstSet)
            out << ", ";

        firstSet = false;

        const ProductionSet &set =
            productions.at(predecessor);

        out << predecessor << ": [";

        for (std::size_t i = 0;
             i < set.alternatives.size();
             ++i) {
            if (i != 0)
                out << ", ";

            const ProductionAlternative &alternative =
                set.alternatives[i];

            out << alternative.probability
                << " -> \""
                << alternative.successor
                << "\"";
        }

        out << "]";
    }

    out << "} ]";
    return out.str();
}

std::optional<TurtleResult> TurtleInterpreter::Interpret(
    const std::string &expanded,
    const LSystemDefinition &definition,
    const FileLoc *loc) {
    TurtleResult result;

    TurtleState state;
    state.branchLength = definition.stepLength;
    state.branchRadius = definition.radius;

    std::vector<TurtleState> stack;
    stack.reserve(64);

    Float angleRadians = DegreesToRadians(definition.angleDegrees);

    std::mt19937 rng(static_cast<std::mt19937::result_type>(definition.seed));
    std::uniform_real_distribution<Float> yawDist(
        -definition.leafYawJitterDegrees,
         definition.leafYawJitterDegrees);
    std::uniform_real_distribution<Float> pitchDist(
        -definition.leafPitchJitterDegrees,
         definition.leafPitchJitterDegrees);
    std::uniform_real_distribution<Float> rollDist(
        -definition.leafRollJitterDegrees,
         definition.leafRollJitterDegrees);

    result.branches.reserve(
        std::min(expanded.size(), definition.maxSymbols));

    for (std::size_t symbolIndex = 0; symbolIndex < expanded.size(); ++symbolIndex) {
        char command = expanded[symbolIndex];

        switch (command) {
        case 'F': {
            Point3f start = state.position;
            Point3f end = start + state.branchLength * state.heading;

            if (!IsFinite(start) || !IsFinite(end)) {
                Error(loc,
                      "Non-finite branch position generated at symbol %d.",
                      int(symbolIndex));
                return std::nullopt;
            }

            int newSegmentIndex = int(result.branches.size());
            result.branches.push_back(
                BranchSegment{
                    start,
                    end,
                    state.branchRadius,
                    state.branchRadius,
                    state.depth,
                    state.parentSegmentIndex,
                    symbolIndex
                });

            state.position = end;
            state.parentSegmentIndex = newSegmentIndex;
            break;
        }

        case 'f':
            state.position += state.branchLength * state.heading;

            if (!IsFinite(state.position)) {
                Error(loc,
                      "Non-finite turtle position generated at symbol %d.",
                      int(symbolIndex));
                return std::nullopt;
            }
            break;

        case '+':
            if (!Yaw(&state, angleRadians, loc, symbolIndex))
                return std::nullopt;
            break;

        case '-':
            if (!Yaw(&state, -angleRadians, loc, symbolIndex))
                return std::nullopt;
            break;

        case '&':
            if (!Pitch(&state, angleRadians, loc, symbolIndex))
                return std::nullopt;
            break;

        case '^':
            if (!Pitch(&state, -angleRadians, loc, symbolIndex))
                return std::nullopt;
            break;

        case '\\':
            if (!Roll(&state, angleRadians, loc, symbolIndex))
                return std::nullopt;
            break;

        case '/':
            if (!Roll(&state, -angleRadians, loc, symbolIndex))
                return std::nullopt;
            break;

        case '|':
            if (!Yaw(&state, Pi, loc, symbolIndex))
                return std::nullopt;
            break;

        case '[':
            stack.push_back(state);

            // The saved state remains at the old depth.  Only the active
            // branch entered after '[' is one level deeper.
            ++state.depth;

            result.maxStackDepth =
                std::max(result.maxStackDepth, stack.size());
            break;

        case ']':
            if (stack.empty()) {
                // Expand() already validates brackets, but keep this check so
                // Interpret() is safe when called directly.
                Error(loc,
                      "Turtle interpreter encountered an unmatched ']' "
                      "at symbol %d.",
                      int(symbolIndex));
                return std::nullopt;
            }

            state = stack.back();
            stack.pop_back();
            break;

        case 'L': {
            TurtleState leafState = state;

            if (!Yaw(&leafState, DegreesToRadians(yawDist(rng)), loc, symbolIndex) ||
                !Pitch(&leafState, DegreesToRadians(pitchDist(rng)), loc, symbolIndex) ||
                !Roll(&leafState, DegreesToRadians(rollDist(rng)), loc, symbolIndex)) {
                return std::nullopt;
            }

            result.leaves.push_back(
                LeafInstance{
                    leafState.position,
                    leafState.heading,
                    leafState.left,
                    leafState.up,
                    definition.leafLength,
                    definition.leafWidth,
                    state.depth,
                    symbolIndex
                });
            break;
        }

        default:
            // Grammar variables such as X and A do not draw anything.
            break;
        }
    }

    if (!stack.empty()) {
        Error(loc,
              "Turtle interpreter finished with %d unclosed branch state(s).",
              int(stack.size()));
        return std::nullopt;
    }

    result.finalState = state;
    return result;
}

}  // namespace pbrt
