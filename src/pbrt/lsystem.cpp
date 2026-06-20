// pbrt L-system deterministic grammar parser
#include <pbrt/lsystem.h>

#include <pbrt/util/error.h>
#include <pbrt/util/math.h>

#include <algorithm>
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

        if (lhs.size() != 1 || !IsValidPredecessor(lhs[0])) {
            Error(loc,
                  "Invalid predecessor \"%s\" in L-system grammar entry %d. "
                  "The first version requires one alphabetic symbol.",
                  lhs.c_str(), int(lineIndex + 1));
            return std::nullopt;
        }

        char predecessor = lhs[0];

        if (definition.productions.find(predecessor) != definition.productions.end()) {
            Error(loc,
                  "Duplicate deterministic production for symbol '%c'.",
                  predecessor);
            return std::nullopt;
        }

        // An empty successor is legal and erases the predecessor.
        definition.productions.emplace(predecessor, std::move(rhs));
    }

    if (!foundAxiom) {
        Error(loc,
              "L-system grammar must contain an axiom declaration such as "
              "\"axiom: F\".");
        return std::nullopt;
    }

    if (definition.axiom.size() > definition.maxSymbols) {
        Error(loc,
              "L-system axiom contains %d symbols, exceeding maxsymbols=%d.",
              int(definition.axiom.size()), int(definition.maxSymbols));
        return std::nullopt;
    }

    return definition;
}

std::optional<std::string> LSystemDefinition::Expand(
    const FileLoc *loc) const {
    std::string current = axiom;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::size_t nextSize = 0;

        // Compute the exact next size before allocating.
        for (char symbol : current) {
            auto iter = productions.find(symbol);
            std::size_t amount =
                iter == productions.end() ? 1 : iter->second.size();

            if (amount > maxSymbols - nextSize) {
                Error(loc,
                      "L-system expansion exceeds maxsymbols=%d while "
                      "computing iteration %d.",
                      int(maxSymbols), iteration + 1);
                return std::nullopt;
            }

            nextSize += amount;
        }

        std::string next;
        next.reserve(nextSize);

        // All replacements read from `current` and write to `next`
        for (char symbol : current) {
            auto iter = productions.find(symbol);
            if (iter == productions.end())
                next.push_back(symbol);
            else
                next.append(iter->second);
        }

        current = std::move(next);
    }

    if (!ValidateBrackets(current, loc))
        return std::nullopt;

    return current;
}

std::string LSystemDefinition::ToString() const {
    std::vector<std::pair<char, std::string>> sortedRules(
        productions.begin(), productions.end());
    std::sort(sortedRules.begin(), sortedRules.end(),
              [](const auto &a, const auto &b) {
                  return a.first < b.first;
              });

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
        << " leafYawJitterDegrees: " << leafYawJitterDegrees
        << " leafPitchJitterDegrees: " << leafPitchJitterDegrees
        << " leafRollJitterDegrees: " << leafRollJitterDegrees
        << " maxSymbols: " << maxSymbols
        << " productions: {";

    bool first = true;
    for (const auto &[predecessor, successor] : sortedRules) {
        if (!first)
            out << ", ";
        first = false;
        out << predecessor << " -> \"" << successor << "\"";
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
