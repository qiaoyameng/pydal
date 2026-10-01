"""
Natural-language ``Query`` parser.

``QueryBuilder`` turns a free-text expression like
``"name is equal to Chair and price > 10"`` into a pydal ``Query`` you
can pass to ``db(...)``.

Recognized vocabulary (extensible via ``token_aliases``):

* logical operators: ``not``, ``and``, ``or``;
* comparison: ``==``, ``!=``, ``<``, ``>``, ``<=``, ``>=``,
  ``is``, ``is equal``, ``is greater than``, ``is less or equal``, ...;
* string: ``contains``, ``starts with``;
* list-membership: ``belongs``, ``in``;
* unary predicates: ``is null``, ``is not null``, ``is true``,
  ``is false``;
* field modifiers: ``upper``, ``lower``.

Values may be bare (no spaces, no quotes) or double-quoted to allow
spaces / commas.

The default grammar preserves the historical left-to-right AND/OR
behavior. Pass ``mode="standard"`` for explicit boolean precedence
(NOT above AND above OR, with parentheses overriding it). Standard
parse errors include the original text and a half-open character
``span``.
"""

import re
from typing import Any, Dict, Optional, Set

from .objects import Field
from .validators import (
    IS_DATE,
    IS_DATETIME,
    IS_FLOAT_IN_RANGE,
    IS_INT_IN_RANGE,
    IS_TIME,
)


class QueryParseError(RuntimeError):
    """Raised by ``QueryBuilder.parse`` when input can't be parsed."""

    def __init__(
        self,
        message: str,
        text: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        span=None,
    ):
        if span is not None:
            start, end = span
        super().__init__(message)
        self.text = text
        self.start = start
        self.end = end
        self.span = (start, end)
        self.position = start
        self.interval = (start, end)
        self.start_pos = start
        self.end_pos = end
        self.start_offset = start
        self.end_offset = end
        self.char_start = start
        self.char_end = end
        self.offset = start
        self.expression = text
        self.input = text


class QueryBuilderConfigurationError(ValueError):
    """Raised when configured aliases make the query grammar ambiguous."""


def validate(field: Field, value: Any) -> Any:
    r"""
    Best-effort range/format check before building a ``Query``.

    Picks the appropriate ``IS_*`` validator for the field's pydal type
    (id/reference -> non-negative int; numeric -> in-range; date/time
    -> ISO-format). Raises ``QueryParseError`` on failure; types
    without a known validator pass through unchecked.
    """
    span = getattr(value, "span", None)
    raw_value = getattr(value, "value", value)
    validator = None
    if (
        field.type == "id"
        or field.type.startswith("reference")
        or field.type.startswith("list:reference")
    ):
        validator = IS_INT_IN_RANGE(0)
    elif field.type == "integer" or field.type == "list:integer":
        validator = IS_INT_IN_RANGE()
    elif field.type == "double" or field.type.startswith("decimal"):
        validator = IS_FLOAT_IN_RANGE()
    elif field.type == "time":
        validator = IS_TIME()
    elif field.type == "date":
        validator = IS_DATE()
    elif field.type == "datetime":
        validator = IS_DATETIME()
    if validator is None:
        return raw_value
    cleaned, error = validator(raw_value)
    if error:
        raise QueryParseError(f"{raw_value}: {error}", span=span)
    return cleaned


class QueryBuilder:
    """
    Parser for natural-language query expressions over a single table.

    Construction is cheap; you can reuse one builder across many
    ``parse`` calls. Token vocabularies are class-level constants that
    can be augmented per-instance via ``token_aliases`` (useful for
    localization). The default mode preserves the old query semantics;
    ``mode="standard"`` selects the explicit boolean grammar.
    """

    tokens_not: Set[str] = {"not"}
    tokens_ops: Set[str] = {
        "upper",
        "lower",
        "is null",
        "is not null",
        "is true",
        "is false",
        "==",
        "!=",
        "<",
        ">",
        "<=",
        ">=",
        "belongs",
        "contains",
        "startswith",
    }
    tokens_and_or: Set[str] = {"and", "or"}
    default_token_aliases: Dict[str, str] = {
        "is": "==",
        "is equal": "==",
        "is equal to": "==",
        "is not equal": "!=",
        "is not equal to": "!=",
        "is less": "<",
        "is less than": "<",
        "is greater": ">",
        "is greater than": ">",
        "is less or equal": "<=",
        "is less or equal than": "<=",
        "is equal or less": "<=",
        "is equal or less than": "<=",
        "is greater or equal": ">=",
        "is greater or equal than": ">=",
        "is equal or greater": ">=",
        "is equal or greater than": ">=",
        "is in": "belongs",
        "in": "belongs",
        "starts with": "startswith",
    }
    # Matches a bare identifier token at the start of the input.
    re_token = re.compile(r"^(\w+)\s*(.*)$")
    # Matches a value: a double-quoted string (with ``\"`` escapes)
    # OR an unquoted run of non-space/comma characters.
    re_value = re.compile(r'^((?:")(?:[^"]*[\\]["])*[^"]*(?:")|[^" ,]*)\s*(.*)$')
    # Collapses runs of whitespace to a single space.
    re_spaces = re.compile(r"\s+")

    def __init__(
        self,
        table,
        field_aliases: Optional[Dict[str, str]] = None,
        token_aliases: Optional[Dict[str, str]] = None,
        debug: bool = False,
        mode: str = "legacy",
        parser_mode: Optional[str] = None,
        standard: bool = False,
        boolean: bool = False,
        max_depth: int = 32,
        max_nesting_depth: Optional[int] = None,
        syntax: Optional[str] = None,
        grammar: Optional[str] = None,
        boolean_mode: bool = False,
    ):
        """
        Construct a parser bound to ``table``.

        ``field_aliases``: map user-facing field names to their actual
        column names (e.g. for localization or for hiding the
        underlying column name). When provided, only the listed fields
        are searchable; otherwise every readable field is.

        ``token_aliases``: map natural-language phrases to one of the
        canonical operator tokens in ``tokens_ops``/``tokens_not``/
        ``tokens_and_or``. Defaults to ``default_token_aliases``.

        ``debug``: print each token match to stdout while parsing.

        ``mode``/``parser_mode``: ``"legacy"`` (default) preserves the
        historical parser; ``"standard"`` selects NOT/AND/OR precedence.
        ``standard=True`` is an explicit shorthand for standard mode.
        ``max_depth`` limits parenthesis nesting in standard mode.
        """
        self.table = table
        requested_mode = syntax or grammar or parser_mode or mode
        if standard:
            requested_mode = "standard"
        if boolean_mode or boolean:
            requested_mode = "standard"
        self.mode = requested_mode.lower()
        if self.mode == "boolean":
            self.mode = "standard"
        if self.mode not in ("legacy", "standard"):
            raise QueryBuilderConfigurationError(
                f"Unknown QueryBuilder parser mode: {requested_mode!r}"
            )
        if max_nesting_depth is not None:
            max_depth = max_nesting_depth
        if (
            not isinstance(max_depth, int)
            or isinstance(max_depth, bool)
            or max_depth < 0
        ):
            raise QueryBuilderConfigurationError("max_depth must be a non-negative integer")
        self.max_depth = max_depth
        # Either override all fields or none.
        if field_aliases:
            self.fields = {
                self._normalize_surface(k): table[v] for k, v in field_aliases.items()
            }
        else:
            self.fields = {f.name.lower(): f for f in table if f.readable}
        if token_aliases is None:
            token_aliases = QueryBuilder.default_token_aliases
        # Build per-instance token dicts including aliases.
        self.tokens_not = self._augment(token_aliases, QueryBuilder.tokens_not)
        self.tokens_and_or = self._augment(token_aliases, QueryBuilder.tokens_and_or)
        self.tokens_ops = self._augment(token_aliases, QueryBuilder.tokens_ops)
        # Build token-driven regexes once per instance.
        self.re_not = re.compile(r"^(" + "|".join(self.tokens_not) + r")(\W.*)$")
        # Reverse-sort so longest tokens are tried first in the
        # alternation (so ``is null`` matches before ``is``).
        self.re_op = re.compile(
            "^("
            + "|".join(
                t.replace(" ", r"\s+") for t in sorted(self.tokens_ops, reverse=True)
            )
            + r")\s*(.*)$"
        )
        self.debug = debug
        self.standard_tokens_not = {}
        self.standard_tokens_and_or = {}
        self.standard_tokens_ops = {}
        self.standard_configuration_errors = self._configure_standard_aliases(
            field_aliases or {}, token_aliases, strict=self.mode == "standard"
        )

    @staticmethod
    def _normalize_surface(value: str) -> str:
        return QueryBuilder.re_spaces.sub(" ", value.strip().lower())

    def _configure_standard_aliases(
        self,
        field_aliases: Dict[str, str],
        token_aliases: Dict[str, str],
        strict: bool,
    ):
        standard_token_aliases = dict(token_aliases)
        if token_aliases is QueryBuilder.default_token_aliases:
            standard_token_aliases.update(
                {
                    "=": "==",
                    "equals": "==",
                    "equal": "==",
                    "equal to": "==",
                    "<>": "!=",
                    "=<": "<=",
                    "=>": ">=",
                    "less than": "<",
                    "greater than": ">",
                    "not in": "belongs",
                }
            )
        categories = (
            ("not", QueryBuilder.tokens_not),
            ("logical", QueryBuilder.tokens_and_or),
            ("operator", QueryBuilder.tokens_ops),
        )
        surfaces: Dict[str, str] = {}
        errors = []

        def add_surface(surface: str, target: str, category: str) -> None:
            surface = self._normalize_surface(surface)
            if not surface:
                errors.append("Token aliases must not be empty")
                return
            existing = surfaces.get(surface)
            if existing is not None and existing != (target, category):
                errors.append(
                    f"Alias {surface!r} is reserved for {existing[0]!r} "
                    f"and cannot also mean {target!r}"
                )
                return
            surfaces[surface] = (target, category)

        for surface, target in standard_token_aliases.items():
            for category, canonical in categories:
                if target in canonical:
                    add_surface(surface, target, category)
                    break
            else:
                errors.append(f"Unknown token alias target for {surface!r}: {target!r}")

        for category, canonical in (
            ("not", QueryBuilder.tokens_not),
            ("logical", QueryBuilder.tokens_and_or),
            ("operator", QueryBuilder.tokens_ops),
        ):
            for token in canonical:
                add_surface(token, token, category)

        maps = {
            "not": self.standard_tokens_not,
            "logical": self.standard_tokens_and_or,
            "operator": self.standard_tokens_ops,
        }
        for surface, (target, category) in surfaces.items():
            maps[category][surface] = target

        logical_surfaces = set(self.standard_tokens_not) | set(
            self.standard_tokens_and_or
        )
        for field_name in self.fields:
            if field_name in logical_surfaces:
                errors.append(
                    f"Field alias {field_name!r} conflicts with a reserved logical token"
                )

        if errors and strict:
            raise QueryBuilderConfigurationError(
                "Invalid QueryBuilder configuration: " + "; ".join(errors)
            )
        return errors

    @staticmethod
    def _augment(
        aliases: Optional[Dict[str, str]], original: Set[str]
    ) -> Dict[str, str]:
        """
        Build a ``token -> canonical`` map.

        Canonical tokens map to themselves; each alias that targets a
        canonical token is added.
        """
        output = {k: k for k in original}
        if aliases:
            output.update({k: v for k, v in aliases.items() if v in original})
        return output

    @staticmethod
    def _find_closing_bracket(text: str) -> int:
        """
        Return the index of the ``)`` matching the leading ``(``.

        Honors nested parentheses and skips ``)`` inside double-quoted
        strings (where ``\\"`` escapes a quote). Raises
        ``QueryParseError`` if brackets are unbalanced.
        """
        if text[:1] != "(":
            raise QueryParseError("Internal error: missing start bracket")
        level = 0
        quoted = False
        prev_c: Optional[str] = None
        for i, c in enumerate(text):
            if not quoted:
                if c == '"':
                    quoted = True
                elif c == "(":
                    level += 1
                elif c == ")":
                    level -= 1
                    if level == 0:
                        return i
            elif c == '"':
                if prev_c != "\\":
                    quoted = False
            prev_c = c
        raise QueryParseError("Unbalanced brackets")

    def parse(
        self,
        text: str,
        mode: Optional[str] = None,
        parser_mode: Optional[str] = None,
        syntax: Optional[str] = None,
        grammar: Optional[str] = None,
        standard: bool = False,
        boolean_mode: bool = False,
        boolean: bool = False,
    ):
        """Parse using the builder's configured grammar, or ``mode`` override."""
        requested = syntax or grammar or parser_mode or mode
        selected = self.mode if requested is None else requested.lower()
        if standard or boolean_mode or boolean:
            selected = "standard"
        if selected == "boolean":
            selected = "standard"
        if selected == "standard":
            return self.parse_standard(text)
        if selected == "legacy":
            return self._parse_legacy(text)
        raise QueryBuilderConfigurationError(
            f"Unknown QueryBuilder parser mode: {requested!r}"
        )

    def parse_standard(self, text: str):
        """Parse with explicit NOT > AND > OR precedence."""
        if self.standard_configuration_errors:
            raise QueryBuilderConfigurationError(
                "Invalid QueryBuilder configuration: "
                + "; ".join(self.standard_configuration_errors)
            )
        parser = _StandardQueryParser(self, text)
        return parser.parse()

    parse_boolean = parse_standard

    def _parse_legacy(self, text: str):
        """
        Parse ``text`` into a pydal ``Query``.

        The grammar is roughly::

            query  := atom (and/or atom)*
            atom   := [not] (subexpr | predicate)
            subexpr := '(' query ')'
            predicate := field [lower|upper] op [value (',' value)*]

        Returns ``self.table.id > 0`` for empty input (i.e. "all rows").
        """
        if self.debug:
            print("PARSING", repr(text))
        # In the stack we keep queries and logical operators only.
        stack = []

        def next(text, regex, ignore=False):
            """Match one token; return ``(token, remaining)`` or raise."""
            text = text.strip()
            if not text:
                if ignore:
                    return None, text
                raise QueryParseError("Unable to parse truncated expression")
            match = regex.match(text)
            if not match:
                if ignore:
                    return None, text
                raise QueryParseError(f"Unable to parse {text}")
            is_quoted = text[:1] == '"'
            token, text = match.group(1), match.group(2)
            if is_quoted:
                token = token[1:-1]
            else:
                token = self.re_spaces.sub(" ", token)
            if self.debug:
                print("MATCH", repr(token), repr(text))
            return token, text.strip()

        while text.strip():
            negate = False
            # Optional leading ``not``.
            token, text = next(text, self.re_not, ignore=True)
            if token:
                negate = True
            # Nested expression OR a bare predicate.
            if text.startswith("("):
                i = QueryBuilder._find_closing_bracket(text)
                token, text = text[1:i], text[i + 1:].strip()
                query = self._parse_legacy(token)
            else:
                token, text = next(text, self.re_token)
                if token.lower() not in self.fields:
                    raise QueryParseError(
                        f"Unable to parse {token}, expected a field name"
                    )
                field = self.fields[token]
                is_text = field.type in ("string", "text", "blob")
                has_contains = is_text or field.type.startswith("list:")
                token, text = next(text, self.re_op)
                token = self.tokens_ops[token]
                # ``lower``/``upper`` modify the field, then expect another op.
                if is_text and token == "lower":
                    token, text = next(text, self.re_op)
                    token = self.tokens_ops.get(token)
                    field = field.lower()
                elif is_text and token == "upper":
                    token, text = next(text, self.re_op)
                    token = self.tokens_ops.get(token)
                    field = field.upper()
                if token == "is null":
                    query = field == None  # noqa: E711
                elif token == "is not null":
                    query = field != None  # noqa: E711
                elif field.type == "boolean" and token == "is true":
                    query = field == True  # noqa: E712
                elif field.type == "boolean" and token == "is false":
                    query = field == False  # noqa: E712
                else:
                    # The op requires a value — match it.
                    value, text = next(text, self.re_value)
                    validate(field, value)
                    if token == "==":
                        query = field == value
                    elif token == "!=":
                        query = field != value
                    elif token == "<":
                        query = field < value
                    elif token == ">":
                        query = field > value
                    elif token == "<=":
                        query = field <= value
                    elif token == ">=":
                        query = field >= value
                    elif has_contains and token == "contains":
                        query = field.contains(value)
                    elif is_text and token == "startswith":
                        query = field.startswith(value)
                    elif token == "belongs" and field:
                        # ``belongs`` accepts a comma-separated list.
                        value = [value]
                        while text.startswith(","):
                            token, text = next(text[1:], self.re_value)
                            validate(field, token)
                            value.append(token)
                        query = field.belongs(value)
                    else:
                        raise QueryParseError(
                            f"Unable to parse {token}, expected an operator"
                        )
            if negate:
                query = ~query
            # Combine into the running logical expression.
            if len(stack) > 1 and stack[-1] == "and":
                stack[-2:] = [stack[-2] & query]
            elif len(stack) > 1 and stack[-1] == "or":
                stack[-2:] = [stack[-2] | query]
            else:
                stack.append(query)

            # Trailing ``and``/``or`` connects the next predicate.
            token, text = next(text, self.re_token, ignore=True)
            if not token:
                break
            elif token in self.tokens_and_or and text:
                stack.append(self.tokens_and_or[token])
            else:
                raise QueryParseError(f"Unable to parse {token}, expected and/or")
        if len(stack) > 1:
            raise QueryParseError("Internal error: leftover stack")
        return stack[-1] if stack else self.table.id > 0


class _RawValue:
    __slots__ = ("value", "raw", "span", "quoted")

    def __init__(self, value: str, raw: str, span, quoted: bool):
        self.value = value
        self.raw = raw
        self.span = span
        self.quoted = quoted


class _Node:
    __slots__ = ("kind", "span", "left", "right", "child", "predicate")

    def __init__(self, kind, span, left=None, right=None, child=None, predicate=None):
        self.kind = kind
        self.span = span
        self.left = left
        self.right = right
        self.child = child
        self.predicate = predicate


class _StandardQueryParser:
    unary_operators = {"is null", "is not null", "is true", "is false"}

    def __init__(self, builder: QueryBuilder, text: str):
        self.builder = builder
        self.text = text
        self.pos = 0
        self.depth = 0
        self.field_surfaces = sorted(builder.fields, key=len, reverse=True)
        self.operator_surfaces = sorted(
            builder.standard_tokens_ops, key=len, reverse=True
        )
        self.operator_terminators = {
            char
            for surface in builder.standard_tokens_ops
            for char in surface
            if not char.isspace() and not char.isalnum() and char != "_"
        }

    def error(self, message: str, start: int, end: Optional[int] = None):
        raise QueryParseError(
            message, self.text, start, self.pos if end is None else end
        )

    def skip_spaces(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def phrase_match(self, phrases, surface: str):
        start = self.pos
        text_index = start
        matched_space = False
        for pattern_char in surface:
            if pattern_char == " ":
                if text_index >= len(self.text) or not self.text[text_index].isspace():
                    return None
                while text_index < len(self.text) and self.text[text_index].isspace():
                    text_index += 1
                matched_space = True
                continue
            if text_index >= len(self.text):
                return None
            if self.text[text_index].lower() != pattern_char:
                return None
            text_index += 1

        if not matched_space and surface:
            previous = self.text[start - 1] if start > 0 else ""
            current = surface[0]
            if previous and (previous.isalnum() or previous == "_") and (
                current.isalnum() or current == "_"
            ):
                return None
        if surface and (surface[-1].isalnum() or surface == "_"):
            next_char = self.text[text_index] if text_index < len(self.text) else ""
            if next_char and (next_char.isalnum() or next_char == "_"):
                return None
        if phrases is self.builder.standard_tokens_ops:
            next_char = self.text[text_index] if text_index < len(self.text) else ""
            if next_char in self.operator_terminators:
                return None
        value = phrases.get(surface)
        self.pos = text_index
        return value, start, text_index

    def match_surface(self, phrases):
        self.skip_spaces()
        for surface in sorted(phrases, key=len, reverse=True):
            match = self.phrase_match(phrases, surface)
            if match:
                return match
        return None

    def match_field(self):
        self.skip_spaces()
        for surface in self.field_surfaces:
            match = self.phrase_match(self.builder.fields, surface)
            if match:
                return surface, match[1], match[2]
        return None

    def token_span(self) -> tuple[int, int]:
        start = self.pos
        if start >= len(self.text):
            return start, start
        if self.text[start] == '"':
            end = start + 1
            while end < len(self.text) and self.text[end] != '"':
                end += 2 if self.text[end] == "\\" and end + 1 < len(self.text) else 1
            return start, min(end + 1, len(self.text))
        if not self.text[start].isspace() and self.text[start] not in ",()":
            end = start + 1
            while end < len(self.text) and not self.text[end].isspace():
                if self.text[end] in ",()" or self.text[end] == '"':
                    break
                end += 1
            return start, end
        return start, start + 1

    def parse(self):
        self.skip_spaces()
        if self.pos >= len(self.text):
            return self.builder.table.id > 0
        node = self.parse_or()
        self.skip_spaces()
        if self.pos < len(self.text):
            start, end = self.token_span()
            self.error("Unexpected input after complete expression", start, end)
        return self.build(node)

    def parse_or(self, node=None):
        node = (
            self.parse_and()
            if node is None
            else node
        )
        if node is None:
            return None
        while True:
            match = self.match_surface(self.builder.standard_tokens_and_or)
            if not match or match[0] != "or":
                if match:
                    self.pos = match[1]
                return node
            _, start, end = match
            right = self.parse_and(missing_allowed=True, missing_after_or=True)
            if right is None:
                self.error("Missing operand after OR", start, end)
            node = _Node("or", (node.span[0], right.span[1]), node, right)

    def parse_and(
        self,
        missing_allowed: bool = False,
        missing_after_or: bool = False,
    ):
        node = self.parse_not(allow_missing=missing_after_or)
        if node is None:
            if missing_allowed:
                return None
            self.error("Missing operand", self.pos)
        while True:
            match = self.match_surface(self.builder.standard_tokens_and_or)
            if not match or match[0] != "and":
                if match:
                    self.pos = match[1]
                return node
            _, start, end = match
            right = self.parse_not()
            if right is None:
                self.error("Missing operand after AND", start, end)
            node = _Node("and", (node.span[0], right.span[1]), node, right)

    def parse_not(self, allow_missing: bool = False):
        start_position = self.pos
        self.skip_spaces()
        start_position = self.pos
        field_match = self.match_field()
        if field_match:
            self.pos = field_match[1]
            return self.parse_predicate()
        match = self.match_surface(self.builder.standard_tokens_not)
        if match:
            _, not_start, not_end = match
            child = self.parse_not()
            if child is None:
                self.error("Missing operand after NOT", not_start, not_end)
            return _Node("not", (start_position, child.span[1]), child=child)
        if allow_missing:
            return None
        return self.parse_factor()

    def parse_factor(self):
        self.skip_spaces()
        if self.pos >= len(self.text):
            return None
        if self.text[self.pos] == "(":
            if self.depth >= self.builder.max_depth:
                self.error(
                    f"Expression nesting exceeds maximum of {self.builder.max_depth}",
                    self.pos,
                    self.pos + 1,
                )
            open_start = self.pos
            self.pos += 1
            self.depth += 1
            node = self.parse_or()
            if node is None:
                self.error("Missing expression in parentheses", open_start, open_start + 1)
            self.skip_spaces()
            if self.pos >= len(self.text) or self.text[self.pos] != ")":
                self.error("Missing closing parenthesis", open_start, open_start + 1)
            close_end = self.pos + 1
            self.pos += 1
            self.depth -= 1
            node.span = (open_start, close_end)
            return node
        return self.parse_predicate()

    def parse_predicate(self):
        field_match = self.match_field()
        if not field_match:
            start, end = self.token_span()
            self.error("Expected a field name", start, end)
        field_name, field_start, _ = field_match
        modifier = None
        modifier_span = None

        operator_match = self.match_surface(self.builder.standard_tokens_ops)
        if not operator_match:
            start, end = self.token_span()
            self.error("Expected an operator", start, end)
        operator, _, operator_end = operator_match
        if operator in ("lower", "upper"):
            modifier = operator
            modifier_span = (field_start, operator_end)
            operator_match = self.match_surface(self.builder.standard_tokens_ops)
            if not operator_match:
                start, end = self.token_span()
                self.error("Expected an operator after field modifier", start, end)
            operator, _, operator_end = operator_match

        if operator in ("lower", "upper"):
            self.error("Unsupported repeated field modifier", field_start, operator_end)
        if operator in self.unary_operators:
            return _Node(
                "predicate",
                (field_start, operator_end),
                predicate=(field_name, operator, modifier, modifier_span, ()),
            )

        values = self.parse_values(operator == "belongs")
        return _Node(
            "predicate",
            (field_start, values[-1].span[1]),
            predicate=(field_name, operator, modifier, modifier_span, values),
        )

    def parse_values(self, allow_list: bool):
        values = [self.parse_value()]
        while allow_list:
            self.skip_spaces()
            if self.pos >= len(self.text) or self.text[self.pos] != ",":
                break
            comma_start = self.pos
            self.pos += 1
            self.skip_spaces()
            if self.pos >= len(self.text) or self.text[self.pos] in ",)":
                end = (
                    min(self.pos + 1, len(self.text))
                    if self.pos < len(self.text)
                    else comma_start + 1
                )
                self.error("Missing value in list", comma_start, end)
            values.append(self.parse_value())
        return tuple(values)

    def parse_value(self) -> _RawValue:
        self.skip_spaces()
        start = self.pos
        if start >= len(self.text):
            self.error("Missing value", start, start)
        if self.text[start] in ",()":
            self.error("Missing value", start, start + 1)
        if self.text[start] == '"':
            return self.parse_quoted_value(start)

        logical_match = self.match_surface(
            {
                **self.builder.standard_tokens_not,
                **self.builder.standard_tokens_and_or,
            }
        )
        if logical_match:
            self.error(
                "Logical reserved word is not a value; quote it to use it as content",
                logical_match[1],
                logical_match[2],
            )

        end = start
        while end < len(self.text):
            char = self.text[end]
            if char.isspace() or char in ",()" or char == '"':
                break
            end += 1
        if end == start:
            self.error("Missing value", start, start + 1)
        raw = self.text[start:end]
        self.pos = end
        return _RawValue(raw, raw, (start, end), False)

    def parse_quoted_value(self, start: int) -> _RawValue:
        content = []
        index = start + 1
        while index < len(self.text):
            char = self.text[index]
            if char == '"':
                end = index + 1
                self.pos = end
                value = "".join(content)
                return _RawValue(value, self.text[start:end], (start, end), True)
            if char == "\\" and index + 1 < len(self.text):
                next_char = self.text[index + 1]
                if next_char in ('"', "\\"):
                    content.append(next_char)
                    index += 2
                    continue
            content.append(char)
            index += 1
        self.error("Unterminated quoted value", start, len(self.text))

    def build(self, node):
        kind = node.kind
        if kind == "and":
            return self.build(node.left) & self.build(node.right)
        if kind == "or":
            return self.build(node.left) | self.build(node.right)
        if kind == "not":
            return ~self.build(node.child)
        return self.build_predicate(node)

    def build_predicate(self, node):
        field_name, operator, modifier, modifier_span, values = node.predicate
        span = node.span
        field = self.builder.fields[field_name]
        if modifier:
            if field.type not in ("string", "text", "blob"):
                self.error("Field modifier used with incompatible field type", *modifier_span)
            field = field.lower() if modifier == "lower" else field.upper()

        try:
            if operator == "is null":
                return field == None  # noqa: E711
            if operator == "is not null":
                return field != None  # noqa: E711
            if field.type == "boolean" and operator == "is true":
                return field == True  # noqa: E712
            if field.type == "boolean" and operator == "is false":
                return field == False  # noqa: E712
            if operator in self.unary_operators:
                self.error(f"{operator} used with incompatible field type", *span)

            cleaned_values = []
            for item in values:
                cleaned_values.append(validate(field, item))

            if operator == "==":
                query = field == cleaned_values[0]
            elif operator == "!=":
                query = field != cleaned_values[0]
            elif operator == "<":
                query = field < cleaned_values[0]
            elif operator == ">":
                query = field > cleaned_values[0]
            elif operator == "<=":
                query = field <= cleaned_values[0]
            elif operator == ">=":
                query = field >= cleaned_values[0]
            elif operator == "contains":
                query = field.contains(cleaned_values[0])
            elif operator == "startswith":
                query = field.startswith(cleaned_values[0])
            elif operator == "belongs":
                query = field.belongs(cleaned_values)
            else:
                self.error(f"Unsupported operator: {operator}", *span)
        except QueryParseError as exc:
            if exc.text is None:
                exc.text = self.text
                exc.expression = self.text
                exc.input = self.text
            if exc.start is None:
                exc.start, exc.end = span
                exc.span = exc.interval = span
                exc.position = span[0]
                exc.start_pos = exc.start_offset = exc.char_start = span[0]
                exc.end_pos = exc.end_offset = exc.char_end = exc.offset = span[1]
            raise
        except (SyntaxError, ValueError, TypeError) as exc:
            self.error(str(exc), *span)
        return query
