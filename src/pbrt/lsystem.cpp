// pbrt parametric deterministic/stochastic L-system parser,
// module-sequence rewriting, 3D turtle interpreter, and branch generation.
#include <pbrt/lsystem.h>

#include <pbrt/util/error.h>
#include <pbrt/util/math.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pbrt {
namespace {

constexpr Float FrameEpsilon = 1e-12f;
constexpr Float ProbabilityNormalizationTolerance = 1e-4f;

using Environment = std::unordered_map<std::string, Float>;

// ---------------------------------------------------------------------------
// General string helpers
// ---------------------------------------------------------------------------

std::string Trim(std::string s) {
    auto isNotSpace = [](unsigned char c) {
        return !std::isspace(c);
    };

    auto begin =
        std::find_if(s.begin(), s.end(), isNotSpace);
    auto end =
        std::find_if(s.rbegin(), s.rend(), isNotSpace).base();

    if (begin >= end)
        return {};

    return std::string(begin, end);
}

std::string ToLower(std::string s) {
    for (char &c : s) {
        c = static_cast<char>(
            std::tolower(static_cast<unsigned char>(c)));
    }

    return s;
}

bool IsIdentifierStart(char c) {
    return std::isalpha(static_cast<unsigned char>(c)) ||
           c == '_';
}

bool IsIdentifierContinue(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) ||
           c == '_';
}

bool IsValidModuleSymbol(char c) {
    // Alphabetic symbols are grammar modules. The remaining symbols are the
    // turtle commands supported by this implementation.
    if (std::isalpha(static_cast<unsigned char>(c)))
        return true;

    const std::string commands = "+-&^\\/|[]!$";
    return commands.find(c) != std::string::npos;
}

bool IsZeroOrOneParameterCommand(char symbol) {
    switch (symbol) {
    case 'F':
    case 'f':
    case '+':
    case '-':
    case '&':
    case '^':
    case '\\':
    case '/':
        return true;
    default:
        return false;
    }
}

bool IsExactlyOneParameterCommand(char symbol) {
    return symbol == '!';
}

bool IsZeroParameterCommand(char symbol) {
    switch (symbol) {
    case '|':
    case '[':
    case ']':
    case '$':
        return true;
    default:
        return false;
    }
}

// F and f are drawing/movement commands rather than ordinary grammar
// variables. All other alphabetic symbols are treated as variables whose
// parameter count must remain fixed throughout the complete L-system.
bool IsGrammarVariable(char symbol) {
    return std::isalpha(static_cast<unsigned char>(symbol)) &&
           symbol != 'F' && symbol != 'f';
}

bool ValidateAndRegisterModuleArity(
    char symbol,
    std::size_t arity,
    std::unordered_map<char, std::size_t> *variableArities,
    const FileLoc *loc,
    const std::string &description) {
    if (IsZeroOrOneParameterCommand(symbol)) {
        if (arity <= 1)
            return true;

        Error(loc,
              "Turtle module '%c' in %s accepts zero parameters "
              "(using the default value) or one explicit parameter; got %d.",
              symbol,
              description.c_str(),
              int(arity));
        return false;
    }

    if (IsExactlyOneParameterCommand(symbol)) {
        if (arity == 1)
            return true;

        Error(loc,
              "Turtle module '%c' in %s requires exactly one parameter; "
              "got %d.",
              symbol,
              description.c_str(),
              int(arity));
        return false;
    }

    if (IsZeroParameterCommand(symbol)) {
        if (arity == 0)
            return true;

        Error(loc,
              "Turtle module '%c' in %s does not accept parameters; got %d.",
              symbol,
              description.c_str(),
              int(arity));
        return false;
    }

    if (!IsGrammarVariable(symbol)) {
        Error(loc,
              "Internal L-system error: no arity rule exists for module '%c' "
              "in %s.",
              symbol,
              description.c_str());
        return false;
    }

    auto [iter, inserted] =
        variableArities->try_emplace(symbol, arity);

    if (inserted || iter->second == arity)
        return true;

    Error(loc,
          "Grammar variable '%c' is used with %d parameter(s) in %s, but "
          "it was previously used with %d parameter(s). Each grammar "
          "variable must have one fixed parameter count throughout the "
          "axiom, predecessors, and successors.",
          symbol,
          int(arity),
          description.c_str(),
          int(iter->second));
    return false;
}

template <typename ModuleLike>
bool ValidateAndRegisterModuleSequenceArities(
    const std::vector<ModuleLike> &modules,
    std::unordered_map<char, std::size_t> *variableArities,
    const FileLoc *loc,
    const std::string &description) {
    for (std::size_t i = 0; i < modules.size(); ++i) {
        std::ostringstream moduleDescription;
        moduleDescription
            << description
            << " at module "
            << i;

        std::size_t arity;

        if constexpr (std::is_same_v<ModuleLike, Module>) {
            arity = modules[i].parameters.size();
        } else {
            arity = modules[i].parameterExpressions.size();
        }

        if (!ValidateAndRegisterModuleArity(
                modules[i].symbol,
                arity,
                variableArities,
                loc,
                moduleDescription.str())) {
            return false;
        }
    }

    return true;
}

std::size_t FindTopLevelCharacter(
    std::string_view text, char target) {
    int parenthesisDepth = 0;

    for (std::size_t i = 0; i < text.size(); ++i) {
        char c = text[i];

        if (c == '(')
            ++parenthesisDepth;
        else if (c == ')')
            --parenthesisDepth;
        else if (c == target && parenthesisDepth == 0)
            return i;

        if (parenthesisDepth < 0)
            return std::string_view::npos;
    }

    return std::string_view::npos;
}

std::vector<std::string> SplitTopLevelArguments(
    std::string_view text, const FileLoc *loc,
    const char *description) {
    std::vector<std::string> arguments;

    if (Trim(std::string(text)).empty())
        return arguments;

    int parenthesisDepth = 0;
    std::size_t begin = 0;

    for (std::size_t i = 0; i <= text.size(); ++i) {
        bool atEnd = i == text.size();
        char c = atEnd ? '\0' : text[i];

        if (!atEnd) {
            if (c == '(')
                ++parenthesisDepth;
            else if (c == ')')
                --parenthesisDepth;

            if (parenthesisDepth < 0) {
                Error(loc,
                      "Unmatched ')' while parsing %s.",
                      description);
                return {};
            }
        }

        if (atEnd || (c == ',' && parenthesisDepth == 0)) {
            std::string argument =
                Trim(std::string(text.substr(begin, i - begin)));

            if (argument.empty()) {
                Error(loc,
                      "Empty parameter while parsing %s.",
                      description);
                return {};
            }

            arguments.push_back(std::move(argument));
            begin = i + 1;
        }
    }

    if (parenthesisDepth != 0) {
        Error(loc,
              "Unbalanced parentheses while parsing %s.",
              description);
        return {};
    }

    return arguments;
}

// ---------------------------------------------------------------------------
// Arithmetic expression parser
// ---------------------------------------------------------------------------

class ExpressionParser {
  public:
    ExpressionParser(
        std::string_view text,
        const FileLoc *loc,
        std::string description)
        : text(text), loc(loc),
          description(std::move(description)) {}

    std::optional<Expression> Parse() {
        Expression expression;

        SkipWhitespace();

        if (AtEnd()) {
            Error(loc,
                  "Empty arithmetic expression in %s.",
                  description.c_str());
            return std::nullopt;
        }

        if (!ParseAddition(&expression.tokens))
            return std::nullopt;

        SkipWhitespace();

        if (!AtEnd()) {
            Error(loc,
                  "Unexpected character '%c' at position %d in %s.",
                  Peek(), int(position), description.c_str());
            return std::nullopt;
        }

        return expression;
    }

  private:
    bool ParseAddition(
        std::vector<ExpressionToken> *output) {
        if (!ParseMultiplication(output))
            return false;

        while (true) {
            SkipWhitespace();

            char operation = Peek();
            if (operation != '+' && operation != '-')
                break;

            ++position;

            if (!ParseMultiplication(output))
                return false;

            output->push_back(ExpressionToken{
                operation == '+'
                    ? ExpressionToken::Type::Add
                    : ExpressionToken::Type::Subtract,
                0.f,
                {}
            });
        }

        return true;
    }

    bool ParseMultiplication(
        std::vector<ExpressionToken> *output) {
        if (!ParseUnary(output))
            return false;

        while (true) {
            SkipWhitespace();

            char operation = Peek();
            if (operation != '*' && operation != '/')
                break;

            ++position;

            if (!ParsePower(output))
                return false;

            output->push_back(ExpressionToken{
                operation == '*'
                    ? ExpressionToken::Type::Multiply
                    : ExpressionToken::Type::Divide,
                0.f,
                {}
            });
        }

        return true;
    }

    // Right-associative exponentiation. Power binds more tightly than unary
    // minus, so -2^2 is interpreted as -(2^2), while 2^-2 remains valid.
    bool ParsePower(
        std::vector<ExpressionToken> *output) {
        if (!ParsePrimary(output))
            return false;

        SkipWhitespace();

        if (Peek() == '^') {
            ++position;

            if (!ParseUnary(output))
                return false;

            output->push_back(ExpressionToken{
                ExpressionToken::Type::Power,
                0.f,
                {}
            });
        }

        return true;
    }

    bool ParseUnary(
        std::vector<ExpressionToken> *output) {
        SkipWhitespace();

        if (Peek() == '+') {
            ++position;
            return ParseUnary(output);
        }

        if (Peek() == '-') {
            ++position;

            if (!ParseUnary(output))
                return false;

            output->push_back(ExpressionToken{
                ExpressionToken::Type::Negate,
                0.f,
                {}
            });

            return true;
        }

        return ParsePower(output);
    }

    bool ParsePrimary(
        std::vector<ExpressionToken> *output) {
        SkipWhitespace();

        if (AtEnd()) {
            Error(loc,
                  "Unexpected end of arithmetic expression in %s.",
                  description.c_str());
            return false;
        }

        if (Peek() == '(') {
            ++position;

            if (!ParseAddition(output))
                return false;

            SkipWhitespace();

            if (Peek() != ')') {
                Error(loc,
                      "Expected ')' in %s.",
                      description.c_str());
                return false;
            }

            ++position;
            return true;
        }

        if (IsIdentifierStart(Peek())) {
            std::size_t begin = position++;
            while (!AtEnd() &&
                   IsIdentifierContinue(Peek())) {
                ++position;
            }

            output->push_back(ExpressionToken{
                ExpressionToken::Type::Variable,
                0.f,
                std::string(
                    text.substr(begin, position - begin))
            });

            return true;
        }

        std::string remaining(
            text.substr(position));

        const char *begin =
            remaining.c_str();
        char *end = nullptr;

        errno = 0;
        double number =
            std::strtod(begin, &end);

        if (end == begin ||
            errno == ERANGE ||
            !std::isfinite(number)) {
            Error(loc,
                  "Expected a number, variable, or parenthesized "
                  "expression at position %d in %s.",
                  int(position), description.c_str());
            return false;
        }

        position +=
            std::size_t(end - begin);

        output->push_back(ExpressionToken{
            ExpressionToken::Type::Number,
            Float(number),
            {}
        });

        return true;
    }

    void SkipWhitespace() {
        while (!AtEnd() &&
               std::isspace(
                   static_cast<unsigned char>(text[position]))) {
            ++position;
        }
    }

    char Peek() const {
        return AtEnd() ? '\0' : text[position];
    }

    bool AtEnd() const {
        return position >= text.size();
    }

    std::string_view text;
    const FileLoc *loc;
    std::string description;
    std::size_t position = 0;
};

std::optional<Expression> ParseExpression(
    std::string_view text,
    const FileLoc *loc,
    const std::string &description) {
    return ExpressionParser(
               text, loc, description)
        .Parse();
}

// ---------------------------------------------------------------------------
// Parametric module parsing
// ---------------------------------------------------------------------------

std::optional<std::vector<ModuleTemplate>>
ParseModuleTemplates(
    std::string_view text,
    const FileLoc *loc,
    const std::string &description) {
    std::vector<ModuleTemplate> modules;
    std::size_t position = 0;

    while (position < text.size()) {
        while (position < text.size() &&
               std::isspace(
                   static_cast<unsigned char>(text[position]))) {
            ++position;
        }

        if (position == text.size())
            break;

        char symbol = text[position];

        if (!IsValidModuleSymbol(symbol)) {
            Error(loc,
                  "Invalid module symbol '%c' at position %d in %s.",
                  symbol, int(position), description.c_str());
            return std::nullopt;
        }

        ++position;

        ModuleTemplate module;
        module.symbol = symbol;

        while (position < text.size() &&
               std::isspace(
                   static_cast<unsigned char>(text[position]))) {
            ++position;
        }

        if (position < text.size() &&
            text[position] == '(') {
            std::size_t argumentsBegin =
                ++position;
            int depth = 1;

            while (position < text.size() &&
                   depth > 0) {
                if (text[position] == '(')
                    ++depth;
                else if (text[position] == ')')
                    --depth;

                ++position;
            }

            if (depth != 0) {
                Error(loc,
                      "Unclosed parameter list for module '%c' in %s.",
                      symbol, description.c_str());
                return std::nullopt;
            }

            std::size_t argumentsEnd =
                position - 1;
            std::string_view argumentText =
                text.substr(
                    argumentsBegin,
                    argumentsEnd - argumentsBegin);

            if (Trim(std::string(argumentText)).empty()) {
                Error(loc,
                      "Module '%c' has an empty parameter list in %s. "
                      "Use '%c' instead of '%c()'.",
                      symbol,
                      description.c_str(),
                      symbol,
                      symbol);
                return std::nullopt;
            }

            std::vector<std::string> arguments =
                SplitTopLevelArguments(
                    argumentText,
                    loc,
                    description.c_str());

            if (arguments.empty())
                return std::nullopt;

            for (std::size_t argumentIndex = 0;
                 argumentIndex < arguments.size();
                 ++argumentIndex) {
                std::ostringstream argumentDescription;
                argumentDescription
                    << "parameter "
                    << argumentIndex
                    << " of module '"
                    << symbol
                    << "' in "
                    << description;

                std::optional<Expression> expression =
                    ParseExpression(
                        arguments[argumentIndex],
                        loc,
                        argumentDescription.str());

                if (!expression)
                    return std::nullopt;

                module.parameterExpressions.push_back(
                    std::move(*expression));
            }
        }

        modules.push_back(std::move(module));
    }

    return modules;
}

std::optional<ParametricWord>
InstantiateWord(
    const std::vector<ModuleTemplate> &templates,
    const Environment &environment,
    const FileLoc *loc) {
    ParametricWord word;
    word.reserve(templates.size());

    for (const ModuleTemplate &moduleTemplate :
         templates) {
        std::optional<Module> module =
            moduleTemplate.Instantiate(
                environment, loc);

        if (!module)
            return std::nullopt;

        word.push_back(std::move(*module));
    }

    return word;
}

std::optional<ModulePattern> ParseModulePattern(
    std::string_view text,
    const FileLoc *loc,
    const std::string &description) {
    std::string trimmed =
        Trim(std::string(text));

    if (trimmed.empty()) {
        Error(loc,
              "Empty production predecessor in %s.",
              description.c_str());
        return std::nullopt;
    }

    std::size_t position = 0;

    char symbol = trimmed[position];

    if (!IsValidModuleSymbol(symbol)) {
        Error(loc,
              "Invalid predecessor module symbol '%c' in %s.",
              symbol, description.c_str());
        return std::nullopt;
    }

    ++position;

    while (position < trimmed.size() &&
           std::isspace(
               static_cast<unsigned char>(trimmed[position]))) {
        ++position;
    }

    ModulePattern pattern;
    pattern.symbol = symbol;

    if (position == trimmed.size())
        return pattern;

    if (trimmed[position] != '(') {
        Error(loc,
              "Unexpected text after predecessor symbol '%c' in %s.",
              symbol, description.c_str());
        return std::nullopt;
    }

    std::size_t close =
        trimmed.find_last_of(')');

    if (close == std::string::npos ||
        close != trimmed.size() - 1) {
        Error(loc,
              "Malformed predecessor parameter list in %s.",
              description.c_str());
        return std::nullopt;
    }

    std::string_view parameterText(
        trimmed.data() + position + 1,
        close - position - 1);

    if (Trim(std::string(parameterText)).empty()) {
        Error(loc,
              "Predecessor '%c' has an empty formal parameter list "
              "in %s.",
              symbol, description.c_str());
        return std::nullopt;
    }

    std::vector<std::string> names =
        SplitTopLevelArguments(
            parameterText, loc,
            description.c_str());

    if (names.empty())
        return std::nullopt;

    for (std::string &name : names) {
        if (name.empty() ||
            !IsIdentifierStart(name[0]) ||
            !std::all_of(
                name.begin() + 1,
                name.end(),
                IsIdentifierContinue)) {
            Error(loc,
                  "Invalid formal parameter name \"%s\" in %s.",
                  name.c_str(), description.c_str());
            return std::nullopt;
        }

        if (std::find(
                pattern.formalParameters.begin(),
                pattern.formalParameters.end(),
                name) !=
            pattern.formalParameters.end()) {
            Error(loc,
                  "Duplicate formal parameter \"%s\" in %s.",
                  name.c_str(), description.c_str());
            return std::nullopt;
        }

        pattern.formalParameters.push_back(
            std::move(name));
    }

    return pattern;
}

ProductionSet *FindProductionSet(
    std::unordered_map<
        char, std::vector<ProductionSet>> *productions,
    char symbol,
    std::size_t arity) {
    auto iter = productions->find(symbol);

    if (iter == productions->end())
        return nullptr;

    for (ProductionSet &set : iter->second) {
        if (set.arity == arity)
            return &set;
    }

    return nullptr;
}

const ProductionSet *FindProductionSet(
    const std::unordered_map<
        char, std::vector<ProductionSet>> &productions,
    char symbol,
    std::size_t arity) {
    auto iter = productions.find(symbol);

    if (iter == productions.end())
        return nullptr;

    for (const ProductionSet &set : iter->second) {
        if (set.arity == arity)
            return &set;
    }

    return nullptr;
}

// ---------------------------------------------------------------------------
// Stochastic production helpers
// ---------------------------------------------------------------------------

bool ParsePositiveWeight(
    const std::string &text,
    Float *weight) {
    if (text.empty())
        return false;

    errno = 0;
    char *end = nullptr;
    double value =
        std::strtod(text.c_str(), &end);

    if (end == text.c_str() ||
        end == nullptr ||
        *end != '\0' ||
        errno == ERANGE ||
        !std::isfinite(value) ||
        value <= 0.0) {
        return false;
    }

    *weight = Float(value);
    return true;
}

Float Uniform01(std::mt19937 *rng) {
    constexpr double InvTwoTo32 =
        1.0 / 4294967296.0;

    return Float(
        double((*rng)()) *
        InvTwoTo32);
}

const ProductionAlternative &ChooseAlternative(
    const ProductionSet &set,
    std::mt19937 *rng) {
    if (!set.stochastic)
        return set.alternatives.front();

    Float u = Uniform01(rng);
    Float cumulative = 0.f;

    for (const ProductionAlternative &alternative :
         set.alternatives) {
        cumulative += alternative.probability;

        if (u < cumulative)
            return alternative;
    }

    return set.alternatives.back();
}

// ---------------------------------------------------------------------------
// Bracket validation
// ---------------------------------------------------------------------------

bool ValidateBrackets(
    const ParametricWord &word,
    const FileLoc *loc,
    const char *description) {
    int depth = 0;

    for (std::size_t i = 0;
         i < word.size();
         ++i) {
        if (word[i].symbol == '[') {
            ++depth;
        } else if (word[i].symbol == ']') {
            --depth;

            if (depth < 0) {
                Error(loc,
                      "%s has an unmatched ']' at module %d.",
                      description, int(i));
                return false;
            }
        }
    }

    if (depth != 0) {
        Error(loc,
              "%s has %d unmatched '[' module(s).",
              description, depth);
        return false;
    }

    return true;
}

bool ValidateTemplateBrackets(
    const std::vector<ModuleTemplate> &modules,
    const FileLoc *loc,
    const char *description) {
    int depth = 0;

    for (std::size_t i = 0;
         i < modules.size();
         ++i) {
        if (modules[i].symbol == '[') {
            ++depth;
        } else if (modules[i].symbol == ']') {
            --depth;

            if (depth < 0) {
                Error(loc,
                      "%s has an unmatched ']' at module %d.",
                      description, int(i));
                return false;
            }
        }
    }

    if (depth != 0) {
        Error(loc,
              "%s has %d unmatched '[' module(s).",
              description, depth);
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Turtle helpers
// ---------------------------------------------------------------------------

Float DegreesToRadians(Float degrees) {
    return degrees * Pi / 180.f;
}

Vector3f RotateAroundAxis(
    Vector3f vector,
    Vector3f axis,
    Float radians) {
    axis = Normalize(axis);

    Float cosTheta = std::cos(radians);
    Float sinTheta = std::sin(radians);

    return cosTheta * vector +
           sinTheta * Cross(axis, vector) +
           (1 - cosTheta) *
               Dot(axis, vector) * axis;
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

bool Reorthonormalize(
    TurtleState *state,
    const FileLoc *loc,
    std::size_t moduleIndex) {
    if (!IsFinite(state->heading) ||
        !IsFinite(state->left) ||
        !IsFinite(state->up) ||
        LengthSquared(state->heading) <=
            FrameEpsilon) {
        Error(loc,
              "Invalid turtle frame after module %d.",
              int(moduleIndex));
        return false;
    }

    state->heading =
        Normalize(state->heading);

    state->left =
        state->left -
        Dot(state->left, state->heading) *
            state->heading;

    if (!IsFinite(state->left) ||
        LengthSquared(state->left) <=
            FrameEpsilon) {
        Error(loc,
              "Degenerate turtle left vector after module %d.",
              int(moduleIndex));
        return false;
    }

    state->left =
        Normalize(state->left);
    state->up =
        Cross(state->heading, state->left);

    if (!IsFinite(state->up) ||
        LengthSquared(state->up) <=
            FrameEpsilon) {
        Error(loc,
              "Degenerate turtle up vector after module %d.",
              int(moduleIndex));
        return false;
    }

    state->up =
        Normalize(state->up);
    state->left =
        Normalize(
            Cross(state->up, state->heading));

    return true;
}

bool Yaw(
    TurtleState *state,
    Float radians,
    const FileLoc *loc,
    std::size_t moduleIndex) {
    state->heading =
        RotateAroundAxis(
            state->heading,
            state->up,
            radians);

    state->left =
        RotateAroundAxis(
            state->left,
            state->up,
            radians);

    return Reorthonormalize(
        state, loc, moduleIndex);
}

bool Pitch(
    TurtleState *state,
    Float radians,
    const FileLoc *loc,
    std::size_t moduleIndex) {
    state->heading =
        RotateAroundAxis(
            state->heading,
            state->left,
            radians);

    state->up =
        RotateAroundAxis(
            state->up,
            state->left,
            radians);

    return Reorthonormalize(
        state, loc, moduleIndex);
}

bool Roll(
    TurtleState *state,
    Float radians,
    const FileLoc *loc,
    std::size_t moduleIndex) {
    state->left =
        RotateAroundAxis(
            state->left,
            state->heading,
            radians);

    state->up =
        RotateAroundAxis(
            state->up,
            state->heading,
            radians);

    return Reorthonormalize(
        state, loc, moduleIndex);
}


// Implements the '$' command from The Algorithmic Beauty of Plants:
//
//     L = normalize(V x H)
//     U = H x L
//
// V points opposite to gravity. H is unchanged; only the turtle's roll
// around H is corrected. If V and H are parallel, the desired horizontal
// left direction is undefined, so the current frame is preserved.
bool AlignBranchPlaneToHorizontal(
    TurtleState *state,
    Vector3f verticalDirection,
    const FileLoc *loc,
    std::size_t moduleIndex) {
    Vector3f horizontalLeft =
        Cross(verticalDirection, state->heading);

    if (!IsFinite(horizontalLeft)) {
        Error(loc,
              "The '$' command generated a non-finite horizontal "
              "direction at module %d.",
              int(moduleIndex));
        return false;
    }

    if (LengthSquared(horizontalLeft) <= FrameEpsilon)
        return true;

    state->left = Normalize(horizontalLeft);
    state->up = Cross(state->heading, state->left);

    return Reorthonormalize(
        state, loc, moduleIndex);
}

// Applies the book's heuristic tropism after drawing one F segment:
//
//     alpha = e * |H x T|
//
// The rotation axis H x T and positive angle alpha rotate H toward T.
// The full H/L/U frame is rotated together so the turtle retains its roll.
bool ApplyTropism(
    TurtleState *state,
    Vector3f tropismVector,
    Float susceptibility,
    const FileLoc *loc,
    std::size_t moduleIndex) {
    if (susceptibility == 0.f ||
        LengthSquared(tropismVector) <= FrameEpsilon) {
        return true;
    }

    Vector3f axis =
        Cross(state->heading, tropismVector);

    Float axisLengthSquared =
        LengthSquared(axis);

    if (!std::isfinite(axisLengthSquared)) {
        Error(loc,
              "Tropism generated a non-finite rotation axis after "
              "module %d.",
              int(moduleIndex));
        return false;
    }

    // H is parallel or antiparallel to T; the cross-product torque is zero.
    if (axisLengthSquared <= FrameEpsilon)
        return true;

    Float axisLength =
        std::sqrt(axisLengthSquared);

    axis = axis / axisLength;

    Float alpha =
        susceptibility * axisLength;

    if (!std::isfinite(alpha)) {
        Error(loc,
              "Tropism generated a non-finite angle after module %d.",
              int(moduleIndex));
        return false;
    }

    state->heading =
        RotateAroundAxis(
            state->heading,
            axis,
            alpha);

    state->left =
        RotateAroundAxis(
            state->left,
            axis,
            alpha);

    state->up =
        RotateAroundAxis(
            state->up,
            axis,
            alpha);

    return Reorthonormalize(
        state, loc, moduleIndex);
}

std::optional<Float> GetOptionalParameter(
    const Module &module,
    Float defaultValue,
    const FileLoc *loc,
    std::size_t moduleIndex) {
    if (module.parameters.empty())
        return defaultValue;

    if (module.parameters.size() != 1) {
        Error(loc,
              "Turtle module '%c' at module %d expects zero or one "
              "parameter; got %d.",
              module.symbol,
              int(moduleIndex),
              int(module.parameters.size()));
        return std::nullopt;
    }

    Float value = module.parameters[0];

    if (!std::isfinite(value)) {
        Error(loc,
              "Turtle module '%c' at module %d has a non-finite "
              "parameter.",
              module.symbol,
              int(moduleIndex));
        return std::nullopt;
    }

    return value;
}

bool RequireNoParameters(
    const Module &module,
    const FileLoc *loc,
    std::size_t moduleIndex) {
    if (module.parameters.empty())
        return true;

    Error(loc,
          "Turtle module '%c' at module %d does not accept parameters.",
          module.symbol,
          int(moduleIndex));
    return false;
}

}  // namespace

// ---------------------------------------------------------------------------
// Expression methods
// ---------------------------------------------------------------------------

std::optional<Float> Expression::Evaluate(
    const Environment &environment,
    const FileLoc *loc) const {
    std::vector<Float> stack;
    stack.reserve(tokens.size());

    auto popOne =
        [&](Float *value) -> bool {
            if (stack.empty()) {
                Error(loc,
                      "Malformed arithmetic expression: stack underflow.");
                return false;
            }

            *value = stack.back();
            stack.pop_back();
            return true;
        };

    auto popTwo =
        [&](Float *left,
            Float *right) -> bool {
            if (stack.size() < 2) {
                Error(loc,
                      "Malformed arithmetic expression: stack underflow.");
                return false;
            }

            *right = stack.back();
            stack.pop_back();

            *left = stack.back();
            stack.pop_back();

            return true;
        };

    for (const ExpressionToken &token :
         tokens) {
        switch (token.type) {
        case ExpressionToken::Type::Number:
            stack.push_back(token.number);
            break;

        case ExpressionToken::Type::Variable: {
            auto iter =
                environment.find(token.variable);

            if (iter == environment.end()) {
                Error(loc,
                      "Unknown L-system expression variable \"%s\".",
                      token.variable.c_str());
                return std::nullopt;
            }

            stack.push_back(iter->second);
            break;
        }

        case ExpressionToken::Type::Negate: {
            Float value;

            if (!popOne(&value))
                return std::nullopt;

            stack.push_back(-value);
            break;
        }

        case ExpressionToken::Type::Add:
        case ExpressionToken::Type::Subtract:
        case ExpressionToken::Type::Multiply:
        case ExpressionToken::Type::Divide:
        case ExpressionToken::Type::Power: {
            Float left, right;

            if (!popTwo(&left, &right))
                return std::nullopt;

            Float result = 0.f;

            switch (token.type) {
            case ExpressionToken::Type::Add:
                result = left + right;
                break;

            case ExpressionToken::Type::Subtract:
                result = left - right;
                break;

            case ExpressionToken::Type::Multiply:
                result = left * right;
                break;

            case ExpressionToken::Type::Divide:
                if (right == 0.f) {
                    Error(loc,
                          "Division by zero in L-system expression.");
                    return std::nullopt;
                }

                result = left / right;
                break;

            case ExpressionToken::Type::Power:
                result =
                    std::pow(left, right);
                break;

            default:
                break;
            }

            if (!std::isfinite(result)) {
                Error(loc,
                      "L-system arithmetic expression evaluated to a "
                      "non-finite value.");
                return std::nullopt;
            }

            stack.push_back(result);
            break;
        }
        }
    }

    if (stack.size() != 1) {
        Error(loc,
              "Malformed arithmetic expression: expected one result, "
              "got %d.",
              int(stack.size()));
        return std::nullopt;
    }

    return stack.back();
}

std::string Expression::ToString() const {
    std::ostringstream out;
    out << "[RPN";

    for (const ExpressionToken &token :
         tokens) {
        out << " ";

        switch (token.type) {
        case ExpressionToken::Type::Number:
            out << token.number;
            break;
        case ExpressionToken::Type::Variable:
            out << token.variable;
            break;
        case ExpressionToken::Type::Add:
            out << "+";
            break;
        case ExpressionToken::Type::Subtract:
            out << "-";
            break;
        case ExpressionToken::Type::Multiply:
            out << "*";
            break;
        case ExpressionToken::Type::Divide:
            out << "/";
            break;
        case ExpressionToken::Type::Power:
            out << "^";
            break;
        case ExpressionToken::Type::Negate:
            out << "neg";
            break;
        }
    }

    out << "]";
    return out.str();
}

// ---------------------------------------------------------------------------
// Module methods
// ---------------------------------------------------------------------------

std::string Module::ToString() const {
    std::ostringstream out;
    out << symbol;

    if (!parameters.empty()) {
        out << "(";

        for (std::size_t i = 0;
             i < parameters.size();
             ++i) {
            if (i != 0)
                out << ",";

            out << parameters[i];
        }

        out << ")";
    }

    return out.str();
}

std::string ParametricWordToString(
    const ParametricWord &word) {
    std::ostringstream out;

    for (const Module &module : word)
        out << module.ToString();

    return out.str();
}

std::string ModulePattern::ToString() const {
    std::ostringstream out;
    out << symbol;

    if (!formalParameters.empty()) {
        out << "(";

        for (std::size_t i = 0;
             i < formalParameters.size();
             ++i) {
            if (i != 0)
                out << ",";

            out << formalParameters[i];
        }

        out << ")";
    }

    return out.str();
}

std::optional<Module> ModuleTemplate::Instantiate(
    const Environment &environment,
    const FileLoc *loc) const {
    Module module;
    module.symbol = symbol;
    module.parameters.reserve(
        parameterExpressions.size());

    for (const Expression &expression :
         parameterExpressions) {
        std::optional<Float> value =
            expression.Evaluate(
                environment, loc);

        if (!value)
            return std::nullopt;

        module.parameters.push_back(*value);
    }

    return module;
}

std::string ModuleTemplate::ToString() const {
    std::ostringstream out;
    out << symbol;

    if (!parameterExpressions.empty()) {
        out << "(";

        for (std::size_t i = 0;
             i < parameterExpressions.size();
             ++i) {
            if (i != 0)
                out << ",";

            out << parameterExpressions[i].ToString();
        }

        out << ")";
    }

    return out.str();
}

// ---------------------------------------------------------------------------
// Skeleton methods
// ---------------------------------------------------------------------------

std::string BranchSegment::ToString() const {
    std::ostringstream out;
    out << "[ BranchSegment"
        << " start: " << start.ToString()
        << " end: " << end.ToString()
        << " radiusStart: " << radiusStart
        << " radiusEnd: " << radiusEnd
        << " depth: " << depth
        << " parentSegmentIndex: "
        << parentSegmentIndex
        << " sourceModuleIndex: "
        << sourceModuleIndex
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
        << " parentSegmentIndex: "
        << parentSegmentIndex
        << " ]";

    return out.str();
}

std::string TurtleResult::ToString() const {
    std::ostringstream out;
    out << "[ TurtleResult"
        << " branchCount: " << branches.size()
        << " maxStackDepth: "
        << maxStackDepth
        << " finalState: "
        << finalState.ToString()
        << " ]";

    return out.str();
}

// ---------------------------------------------------------------------------
// Definition parsing
// ---------------------------------------------------------------------------

std::optional<LSystemDefinition>
LSystemDefinition::Create(
    const ParameterDictionary &parameters,
    const FileLoc *loc) {
    LSystemDefinition definition;

    std::vector<std::string> grammarLines =
        parameters.GetStringArray("grammar");

    std::vector<std::string> constantLines =
        parameters.GetStringArray("constants");

    definition.iterations =
        parameters.GetOneInt(
            "iterations", 0);

    definition.angleDegrees =
        parameters.GetOneFloat(
            "angle", 25.f);

    definition.stepLength =
        parameters.GetOneFloat(
            "length", 1.f);

    definition.radius =
        parameters.GetOneFloat(
            "radius", 0.05f);

    definition.verticalDirection =
        parameters.GetOneVector3f(
            "vertical",
            Vector3f(0, 1, 0));

    definition.tropismVector =
        parameters.GetOneVector3f(
            "tropism",
            Vector3f(0, 0, 0));

    definition.tropismSusceptibility =
        parameters.GetOneFloat(
            "susceptibility", 0.f);

    definition.seed =
        parameters.GetOneInt(
            "seed", 1);

    definition.grammarSeed =
        parameters.GetOneInt(
            "grammarseed",
            definition.seed);

    int legacyMaxSymbols =
        parameters.GetOneInt(
            "maxsymbols", 1'000'000);

    int maxModules =
        parameters.GetOneInt(
            "maxmodules",
            legacyMaxSymbols);

    if (grammarLines.empty()) {
        Error(loc,
              "Shape \"lsystem\" requires a non-empty "
              "\"string grammar\" parameter.");
        return std::nullopt;
    }

    if (definition.iterations < 0 ||
        definition.iterations > 64) {
        Error(loc,
              "L-system \"iterations\" must be between 0 and 64; "
              "got %d.",
              definition.iterations);
        return std::nullopt;
    }

    if (!std::isfinite(
            definition.angleDegrees)) {
        Error(loc,
              "L-system \"angle\" must be finite; got %f.",
              definition.angleDegrees);
        return std::nullopt;
    }

    if (!std::isfinite(
            definition.stepLength) ||
        definition.stepLength <= 0.f) {
        Error(loc,
              "L-system \"length\" must be finite and greater "
              "than zero; got %f.",
              definition.stepLength);
        return std::nullopt;
    }

    if (!std::isfinite(
            definition.radius) ||
        definition.radius <= 0.f) {
        Error(loc,
              "L-system \"radius\" must be finite and greater "
              "than zero; got %f.",
              definition.radius);
        return std::nullopt;
    }

    if (!IsFinite(
            definition.verticalDirection) ||
        LengthSquared(
            definition.verticalDirection) <=
            FrameEpsilon) {
        Error(loc,
              "L-system \"vertical\" must be a finite, nonzero "
              "vector.");
        return std::nullopt;
    }

    definition.verticalDirection =
        Normalize(
            definition.verticalDirection);

    if (!IsFinite(
            definition.tropismVector)) {
        Error(loc,
              "L-system \"tropism\" must be a finite vector.");
        return std::nullopt;
    }

    if (!std::isfinite(
            definition.tropismSusceptibility) ||
        definition.tropismSusceptibility < 0.f) {
        Error(loc,
              "L-system \"susceptibility\" must be finite and "
              "nonnegative; got %f.",
              definition.tropismSusceptibility);
        return std::nullopt;
    }

    if (maxModules <= 0) {
        Error(loc,
              "L-system \"maxmodules\" must be greater than zero; "
              "got %d.",
              maxModules);
        return std::nullopt;
    }

    definition.maxModules =
        static_cast<std::size_t>(
            maxModules);

    // A built-in constant is useful for rotation and growth expressions.
    definition.constants.emplace(
        "pi", Pi);

    // Parse constants in order. Later constants may use earlier constants.
    for (std::size_t lineIndex = 0;
         lineIndex < constantLines.size();
         ++lineIndex) {
        std::string line =
            constantLines[lineIndex];

        if (std::size_t comment =
                line.find('#');
            comment != std::string::npos) {
            line.erase(comment);
        }

        line = Trim(std::move(line));

        if (line.empty())
            continue;

        std::size_t equal =
            FindTopLevelCharacter(
                line, '=');

        if (equal ==
            std::string::npos) {
            Error(loc,
                  "Invalid L-system constant entry %d: \"%s\". "
                  "Expected \"name = expression\".",
                  int(lineIndex + 1),
                  line.c_str());
            return std::nullopt;
        }

        std::string name =
            Trim(line.substr(0, equal));

        std::string expressionText =
            Trim(line.substr(equal + 1));

        if (name.empty() ||
            !IsIdentifierStart(name[0]) ||
            !std::all_of(
                name.begin() + 1,
                name.end(),
                IsIdentifierContinue)) {
            Error(loc,
                  "Invalid L-system constant name \"%s\".",
                  name.c_str());
            return std::nullopt;
        }

        if (name == "pi") {
            Error(loc,
                  "The built-in L-system constant \"pi\" may not "
                  "be redefined.");
            return std::nullopt;
        }

        if (definition.constants.find(name) !=
            definition.constants.end()) {
            Error(loc,
                  "Duplicate L-system constant \"%s\".",
                  name.c_str());
            return std::nullopt;
        }

        std::optional<Expression> expression =
            ParseExpression(
                expressionText,
                loc,
                "constant \"" + name + "\"");

        if (!expression)
            return std::nullopt;

        std::optional<Float> value =
            expression->Evaluate(
                definition.constants,
                loc);

        if (!value)
            return std::nullopt;

        definition.constants.emplace(
            std::move(name), *value);
    }

    bool foundAxiom = false;

    for (std::size_t lineIndex = 0;
         lineIndex < grammarLines.size();
         ++lineIndex) {
        std::string line =
            grammarLines[lineIndex];

        if (std::size_t comment =
                line.find('#');
            comment != std::string::npos) {
            line.erase(comment);
        }

        line = Trim(std::move(line));

        if (line.empty())
            continue;

        std::size_t arrow =
            line.find("->");

        if (arrow ==
            std::string::npos) {
            std::size_t colon =
                FindTopLevelCharacter(
                    line, ':');

            std::size_t equal =
                FindTopLevelCharacter(
                    line, '=');

            std::size_t separator =
                std::string::npos;

            if (colon !=
                std::string::npos) {
                separator = colon;
            }

            if (equal !=
                    std::string::npos &&
                (separator ==
                     std::string::npos ||
                 equal < separator)) {
                separator = equal;
            }

            if (separator !=
                std::string::npos) {
                std::string key =
                    ToLower(
                        Trim(
                            line.substr(
                                0, separator)));

                if (key == "axiom") {
                    if (foundAxiom) {
                        Error(loc,
                              "L-system grammar contains more than "
                              "one axiom declaration.");
                        return std::nullopt;
                    }

                    std::string axiomText =
                        Trim(
                            line.substr(
                                separator + 1));

                    std::optional<
                        std::vector<ModuleTemplate>>
                        axiomTemplates =
                            ParseModuleTemplates(
                                axiomText,
                                loc,
                                "L-system axiom");

                    if (!axiomTemplates ||
                        axiomTemplates->empty()) {
                        Error(loc,
                              "L-system axiom may not be empty.");
                        return std::nullopt;
                    }

                    std::optional<ParametricWord>
                        axiom =
                            InstantiateWord(
                                *axiomTemplates,
                                definition.constants,
                                loc);

                    if (!axiom)
                        return std::nullopt;

                    if (!ValidateBrackets(
                            *axiom,
                            loc,
                            "L-system axiom")) {
                        return std::nullopt;
                    }

                    if (!ValidateAndRegisterModuleSequenceArities(
                            *axiom,
                            &definition.variableArities,
                            loc,
                            "L-system axiom")) {
                        return std::nullopt;
                    }

                    definition.axiom =
                        std::move(*axiom);

                    foundAxiom = true;
                    continue;
                }
            }

            Error(loc,
                  "Invalid L-system grammar entry %d: \"%s\". "
                  "Expected \"axiom: ...\", "
                  "\"A(x) -> ...\", or "
                  "\"A(x):weight -> ...\". "
                  "Conditions are not supported in this version.",
                  int(lineIndex + 1),
                  line.c_str());
            return std::nullopt;
        }

        std::string lhs =
            Trim(line.substr(0, arrow));

        std::string rhs =
            Trim(line.substr(arrow + 2));

        std::size_t weightSeparator =
            FindTopLevelCharacter(
                lhs, ':');

        bool stochastic =
            weightSeparator !=
            std::string::npos;

        std::string predecessorText =
            stochastic
                ? Trim(
                      lhs.substr(
                          0, weightSeparator))
                : lhs;

        Float weight = 1.f;

        if (stochastic) {
            std::string weightText =
                Trim(
                    lhs.substr(
                        weightSeparator + 1));

            if (!ParsePositiveWeight(
                    weightText,
                    &weight)) {
                Error(loc,
                      "Invalid stochastic weight \"%s\" in grammar "
                      "entry %d. Conditions are not supported; text "
                      "after ':' must be a finite positive number.",
                      weightText.c_str(),
                      int(lineIndex + 1));
                return std::nullopt;
            }
        }

        std::ostringstream productionDescription;
        productionDescription
            << "grammar entry "
            << lineIndex + 1;

        std::optional<ModulePattern>
            predecessor =
                ParseModulePattern(
                    predecessorText,
                    loc,
                    productionDescription.str());

        if (!predecessor)
            return std::nullopt;

        if (!ValidateAndRegisterModuleArity(
                predecessor->symbol,
                predecessor->formalParameters.size(),
                &definition.variableArities,
                loc,
                "production predecessor " +
                    predecessor->ToString())) {
            return std::nullopt;
        }

        std::optional<
            std::vector<ModuleTemplate>>
            successor =
                ParseModuleTemplates(
                    rhs,
                    loc,
                    "successor of " +
                        predecessor->ToString());

        if (!successor)
            return std::nullopt;

        if (!ValidateAndRegisterModuleSequenceArities(
                *successor,
                &definition.variableArities,
                loc,
                "successor of " +
                    predecessor->ToString())) {
            return std::nullopt;
        }

        if (!ValidateTemplateBrackets(
                *successor,
                loc,
                "L-system production successor")) {
            return std::nullopt;
        }

        ProductionSet *set =
            FindProductionSet(
                &definition.productions,
                predecessor->symbol,
                predecessor->
                    formalParameters.size());

        if (!set) {
            ProductionSet newSet;
            newSet.symbol =
                predecessor->symbol;
            newSet.arity =
                predecessor->
                    formalParameters.size();
            newSet.stochastic =
                stochastic;

            definition.productions[
                predecessor->symbol]
                .push_back(
                    std::move(newSet));

            set = &definition.productions[
                predecessor->symbol]
                .back();
        } else if (set->stochastic !=
                   stochastic) {
            Error(loc,
                  "Cannot mix deterministic and stochastic productions "
                  "for module '%c' with arity %d.",
                  predecessor->symbol,
                  int(predecessor->
                          formalParameters.size()));
            return std::nullopt;
        }

        if (!set->stochastic &&
            !set->alternatives.empty()) {
            Error(loc,
                  "Duplicate deterministic production for module '%c' "
                  "with arity %d.",
                  predecessor->symbol,
                  int(predecessor->
                          formalParameters.size()));
            return std::nullopt;
        }

        set->alternatives.push_back(
            ProductionAlternative{
                weight,
                std::move(*predecessor),
                std::move(*successor)
            });
    }

    if (!foundAxiom) {
        Error(loc,
              "L-system grammar must contain an axiom declaration.");
        return std::nullopt;
    }

    // Normalize stochastic alternatives independently for each symbol/arity.
    for (auto &[symbol, sets] :
         definition.productions) {
        for (ProductionSet &set : sets) {
            if (set.alternatives.empty()) {
                Error(loc,
                      "L-system production set for '%c' with arity %d "
                      "is empty.",
                      symbol, int(set.arity));
                return std::nullopt;
            }

            if (!set.stochastic) {
                set.alternatives[0].probability =
                    1.f;
                continue;
            }

            double weightSum = 0.0;

            for (const ProductionAlternative &alternative :
                 set.alternatives) {
                weightSum +=
                    double(
                        alternative.probability);
            }

            if (!std::isfinite(weightSum) ||
                weightSum <= 0.0) {
                Error(loc,
                      "Stochastic weights for module '%c' with "
                      "arity %d have invalid sum %f.",
                      symbol,
                      int(set.arity),
                      weightSum);
                return std::nullopt;
            }

            if (std::abs(
                    weightSum - 1.0) >
                double(
                    ProbabilityNormalizationTolerance)) {
                Warning(loc,
                        "Stochastic weights for module '%c' with "
                        "arity %d sum to %f; normalizing them to one.",
                        symbol,
                        int(set.arity),
                        weightSum);
            }

            Float inverseWeightSum =
                Float(1.0 / weightSum);

            for (ProductionAlternative &alternative :
                 set.alternatives) {
                alternative.probability *=
                    inverseWeightSum;
            }
        }
    }

    if (definition.axiom.size() >
        definition.maxModules) {
        Error(loc,
              "L-system axiom contains %d modules, exceeding "
              "maxmodules=%d.",
              int(definition.axiom.size()),
              int(definition.maxModules));
        return std::nullopt;
    }

    return definition;
}

// ---------------------------------------------------------------------------
// Parallel module-sequence rewriting
// ---------------------------------------------------------------------------

std::optional<ParametricWord>
LSystemDefinition::Expand(
    const FileLoc *loc) const {
    ParametricWord current = axiom;

    std::mt19937 grammarRng(
        static_cast<
            std::mt19937::result_type>(
                grammarSeed));

    for (int iteration = 0;
         iteration < iterations;
         ++iteration) {
        ParametricWord next;
        next.reserve(current.size());

        for (const Module &module :
             current) {
            const ProductionSet *set =
                FindProductionSet(
                    productions,
                    module.symbol,
                    module.parameters.size());

            // Identity production: unmatched modules reproduce themselves.
            if (!set) {
                if (next.size() ==
                    maxModules) {
                    Error(loc,
                          "L-system expansion exceeds maxmodules=%d "
                          "while computing iteration %d.",
                          int(maxModules),
                          iteration + 1);
                    return std::nullopt;
                }

                next.push_back(module);
                continue;
            }

            const ProductionAlternative &alternative =
                ChooseAlternative(
                    *set,
                    &grammarRng);

            Environment environment =
                constants;

            if (alternative.predecessor
                    .formalParameters.size() !=
                module.parameters.size()) {
                Error(loc,
                      "Internal L-system error: predecessor/module "
                      "arity mismatch for '%c'.",
                      module.symbol);
                return std::nullopt;
            }

            for (std::size_t i = 0;
                 i < module.parameters.size();
                 ++i) {
                environment[
                    alternative.predecessor
                        .formalParameters[i]] =
                    module.parameters[i];
            }

            if (alternative.successor.size() >
                maxModules - next.size()) {
                Error(loc,
                      "L-system expansion exceeds maxmodules=%d "
                      "while computing iteration %d.",
                      int(maxModules),
                      iteration + 1);
                return std::nullopt;
            }

            for (const ModuleTemplate &output :
                 alternative.successor) {
                std::optional<Module> generated =
                    output.Instantiate(
                        environment,
                        loc);

                if (!generated)
                    return std::nullopt;

                next.push_back(
                    std::move(*generated));
            }
        }

        current = std::move(next);
    }

    if (!ValidateBrackets(
            current,
            loc,
            "Expanded L-system word")) {
        return std::nullopt;
    }

    return current;
}

std::string LSystemDefinition::ToString() const {
    std::ostringstream out;

    out << "[ LSystemDefinition"
        << " axiom: \""
        << ParametricWordToString(axiom)
        << "\""
        << " iterations: "
        << iterations
        << " angleDegrees: "
        << angleDegrees
        << " stepLength: "
        << stepLength
        << " radius: "
        << radius
        << " verticalDirection: "
        << verticalDirection.ToString()
        << " tropismVector: "
        << tropismVector.ToString()
        << " tropismSusceptibility: "
        << tropismSusceptibility
        << " seed: "
        << seed
        << " grammarSeed: "
        << grammarSeed
        << " maxModules: "
        << maxModules
        << " constants: {";

    std::vector<std::string> constantNames;
    constantNames.reserve(constants.size());

    for (const auto &[name, value] :
         constants) {
        constantNames.push_back(name);
    }

    std::sort(
        constantNames.begin(),
        constantNames.end());

    for (std::size_t i = 0;
         i < constantNames.size();
         ++i) {
        if (i != 0)
            out << ", ";

        out << constantNames[i]
            << "="
            << constants.at(
                   constantNames[i]);
    }

    out << "} variableArities: {";

    std::vector<char> variableSymbols;
    variableSymbols.reserve(variableArities.size());

    for (const auto &[symbol, arity] :
         variableArities) {
        variableSymbols.push_back(symbol);
    }

    std::sort(
        variableSymbols.begin(),
        variableSymbols.end());

    for (std::size_t i = 0;
         i < variableSymbols.size();
         ++i) {
        if (i != 0)
            out << ", ";

        char symbol = variableSymbols[i];
        out << symbol
            << "="
            << variableArities.at(symbol);
    }

    out << "} productions: {";

    std::vector<char> symbols;
    symbols.reserve(productions.size());

    for (const auto &[symbol, sets] :
         productions) {
        symbols.push_back(symbol);
    }

    std::sort(
        symbols.begin(),
        symbols.end());

    bool firstSet = true;

    for (char symbol : symbols) {
        std::vector<const ProductionSet *> sets;

        for (const ProductionSet &set :
             productions.at(symbol)) {
            sets.push_back(&set);
        }

        std::sort(
            sets.begin(),
            sets.end(),
            [](const ProductionSet *a,
               const ProductionSet *b) {
                return a->arity < b->arity;
            });

        for (const ProductionSet *set :
             sets) {
            if (!firstSet)
                out << ", ";

            firstSet = false;

            out << symbol
                << "/"
                << set->arity
                << ": [";

            for (std::size_t i = 0;
                 i < set->alternatives.size();
                 ++i) {
                if (i != 0)
                    out << ", ";

                const ProductionAlternative &alternative =
                    set->alternatives[i];

                out << alternative.probability
                    << " "
                    << alternative.predecessor.ToString()
                    << " -> ";

                for (const ModuleTemplate &module :
                     alternative.successor) {
                    out << module.ToString();
                }
            }

            out << "]";
        }
    }

    out << "} ]";
    return out.str();
}

// ---------------------------------------------------------------------------
// Parametric turtle interpretation
// ---------------------------------------------------------------------------

std::optional<TurtleResult>
TurtleInterpreter::Interpret(
    const ParametricWord &expanded,
    const LSystemDefinition &definition,
    const FileLoc *loc) {
    TurtleResult result;

    TurtleState state;
    state.branchLength =
        definition.stepLength;
    state.branchRadius =
        definition.radius;

    std::vector<TurtleState> stack;
    stack.reserve(64);

    result.branches.reserve(
        expanded.size());

    for (std::size_t moduleIndex = 0;
         moduleIndex < expanded.size();
         ++moduleIndex) {
        const Module &module =
            expanded[moduleIndex];

        switch (module.symbol) {
        case 'F': {
            std::optional<Float> length =
                GetOptionalParameter(
                    module,
                    state.branchLength,
                    loc,
                    moduleIndex);

            if (!length)
                return std::nullopt;

            if (*length < 0.f) {
                Error(loc,
                      "F module at module %d has negative length %f.",
                      int(moduleIndex), *length);
                return std::nullopt;
            }

            Point3f start =
                state.position;

            Point3f end =
                start +
                *length *
                    state.heading;

            if (!IsFinite(start) ||
                !IsFinite(end)) {
                Error(loc,
                      "Non-finite branch position generated at "
                      "module %d.",
                      int(moduleIndex));
                return std::nullopt;
            }

            int newSegmentIndex =
                int(result.branches.size());

            result.branches.push_back(
                BranchSegment{
                    start,
                    end,
                    state.branchRadius,
                    state.branchRadius,
                    state.depth,
                    state.parentSegmentIndex,
                    moduleIndex
                });

            state.position = end;
            state.parentSegmentIndex =
                newSegmentIndex;

            if (*length > 0.f &&
                !ApplyTropism(
                    &state,
                    definition.tropismVector,
                    definition.tropismSusceptibility,
                    loc,
                    moduleIndex)) {
                return std::nullopt;
            }

            break;
        }

        case 'f': {
            std::optional<Float> length =
                GetOptionalParameter(
                    module,
                    state.branchLength,
                    loc,
                    moduleIndex);

            if (!length)
                return std::nullopt;

            if (*length < 0.f) {
                Error(loc,
                      "f module at module %d has negative length %f.",
                      int(moduleIndex), *length);
                return std::nullopt;
            }

            state.position +=
                *length *
                state.heading;

            if (!IsFinite(
                    state.position)) {
                Error(loc,
                      "Non-finite turtle position generated at "
                      "module %d.",
                      int(moduleIndex));
                return std::nullopt;
            }

            break;
        }

        case '+':
        case '-': {
            std::optional<Float> angle =
                GetOptionalParameter(
                    module,
                    definition.angleDegrees,
                    loc,
                    moduleIndex);

            if (!angle)
                return std::nullopt;

            Float signedAngle =
                module.symbol == '+'
                    ? *angle
                    : -*angle;

            if (!Yaw(
                    &state,
                    DegreesToRadians(
                        signedAngle),
                    loc,
                    moduleIndex)) {
                return std::nullopt;
            }

            break;
        }

        case '&':
        case '^': {
            std::optional<Float> angle =
                GetOptionalParameter(
                    module,
                    definition.angleDegrees,
                    loc,
                    moduleIndex);

            if (!angle)
                return std::nullopt;

            Float signedAngle =
                module.symbol == '&'
                    ? *angle
                    : -*angle;

            if (!Pitch(
                    &state,
                    DegreesToRadians(
                        signedAngle),
                    loc,
                    moduleIndex)) {
                return std::nullopt;
            }

            break;
        }

        case '\\':
        case '/': {
            std::optional<Float> angle =
                GetOptionalParameter(
                    module,
                    definition.angleDegrees,
                    loc,
                    moduleIndex);

            if (!angle)
                return std::nullopt;

            Float signedAngle =
                module.symbol == '\\'
                    ? *angle
                    : -*angle;

            if (!Roll(
                    &state,
                    DegreesToRadians(
                        signedAngle),
                    loc,
                    moduleIndex)) {
                return std::nullopt;
            }

            break;
        }

        case '!': {
            if (module.parameters.size() != 1) {
                Error(loc,
                      "Turtle module '!' at module %d requires exactly "
                      "one full-width parameter.",
                      int(moduleIndex));
                return std::nullopt;
            }

            Float width =
                module.parameters[0];

            if (!std::isfinite(width) ||
                width <= 0.f) {
                Error(loc,
                      "Turtle module '!' at module %d requires a "
                      "finite positive width; got %f.",
                      int(moduleIndex),
                      width);
                return std::nullopt;
            }

            state.branchRadius =
                0.5f * width;
            break;
        }

        case '$':
            if (!RequireNoParameters(
                    module,
                    loc,
                    moduleIndex)) {
                return std::nullopt;
            }

            if (!AlignBranchPlaneToHorizontal(
                    &state,
                    definition.verticalDirection,
                    loc,
                    moduleIndex)) {
                return std::nullopt;
            }
            break;

        case '|':
            if (!RequireNoParameters(
                    module,
                    loc,
                    moduleIndex)) {
                return std::nullopt;
            }

            if (!Yaw(
                    &state,
                    Pi,
                    loc,
                    moduleIndex)) {
                return std::nullopt;
            }
            break;

        case '[':
            if (!RequireNoParameters(
                    module,
                    loc,
                    moduleIndex)) {
                return std::nullopt;
            }

            stack.push_back(state);
            ++state.depth;

            result.maxStackDepth =
                std::max(
                    result.maxStackDepth,
                    stack.size());
            break;

        case ']':
            if (!RequireNoParameters(
                    module,
                    loc,
                    moduleIndex)) {
                return std::nullopt;
            }

            if (stack.empty()) {
                Error(loc,
                      "Turtle interpreter encountered an unmatched ']' "
                      "at module %d.",
                      int(moduleIndex));
                return std::nullopt;
            }

            state = stack.back();
            stack.pop_back();
            break;

        default:
            // Parametric grammar variables such as A(l,w) and X(t) do not
            // directly generate geometry.
            break;
        }
    }

    if (!stack.empty()) {
        Error(loc,
              "Turtle interpreter finished with %d unclosed branch "
              "state(s).",
              int(stack.size()));
        return std::nullopt;
    }

    result.finalState = state;
    return result;
}

}  // namespace pbrt
