"""
Natural-language ``Query`` parser.

``QueryBuilder`` turns a free-text expression like
``"name is equal to Chair and price > 10"`` into a pydal ``Query`` you
can pass to ``db(...)``.

Two parsing modes are available:

* ``legacy`` (default): the historical left-to-right grammar, kept
  unchanged so existing expressions keep their meaning;
* ``standard``: an explicit boolean grammar where ``not`` binds
  tighter than ``and``, which binds tighter than ``or``, and
  parentheses override precedence. The whole input is tokenized
  first (fields, operators, values), then a ``Query`` is built, so
  syntax errors carry character offsets into the *original* input,
  even inside nested parentheses.

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

See ``README.md`` for examples; the source-level regexes are the
authoritative grammar.
"""

import re
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

from .objects import Field
from .validators import (
    IS_DATE,
    IS_DATETIME,
    IS_FLOAT_IN_RANGE,
    IS_INT_IN_RANGE,
    IS_TIME,
)


class QueryParseError(RuntimeError):
    """
    Raised by ``QueryBuilder.parse`` when input can't be parsed.

    ``start``/``end`` hold the offending character range within the
    original input string when known (always set in ``standard``
    mode); both are ``None`` for errors without position info.
    """

    def __init__(
        self,
        message: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ):
        if start is not None and end is not None:
            message = f"{message} (characters {start}-{end})"
        super().__init__(message)
        self.start = start
        self.end = end


class QueryBuilderConfigError(RuntimeError):
    """
    Raised by ``QueryBuilder.__init__`` when the configuration is
    inconsistent (e.g. an alias collides with a reserved word).
    """


def validate(field: Field, value: Any) -> None:
    """
    Best-effort range/format check before building a ``Query``.

    Picks the appropriate ``IS_*`` validator for the field's pydal type
    (id/reference -> non-negative int; numeric -> in-range; date/time
    -> ISO-format). Raises ``QueryParseError`` on failure; types
    without a known validator pass through unchecked.
    """
    error = None
    if (
        field.type == "id"
        or field.type.startswith("reference")
        or field.type.startswith("list:reference")
    ):
        error = IS_INT_IN_RANGE(0)(value)[1]
    elif field.type == "integer" or field.type == "list:integer":
        error = IS_INT_IN_RANGE()(value)[1]
    elif field.type == "double" or field.type.startswith("decimal"):
        error = IS_FLOAT_IN_RANGE()(value)[1]
    elif field.type == "time":
        error = IS_TIME()(value)[1]
    elif field.type == "date":
        error = IS_DATE()(value)[1]
    elif field.type == "datetime":
        error = IS_DATETIME()(value)[1]
    if error:
        raise QueryParseError(f"{value}: {error}")


class QueryBuilder:
    """
    Parser for natural-language query expressions over a single table.

    Construction is cheap; you can reuse one builder across many
    ``parse`` calls. Token vocabularies are class-level constants that
    can be augmented per-instance via ``token_aliases`` (useful for
    localization).
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
        max_depth: Optional[int] = 32,
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

        ``mode``: ``"legacy"`` (default) keeps the historical
        left-to-right grammar; ``"standard"`` enables the explicit
        boolean grammar (``not`` > ``and`` > ``or``, parentheses
        override, consecutive ``not`` allowed, position-aware
        errors). The default never changes, so existing expressions
        keep their meaning unless ``standard`` is requested.

        ``max_depth``: maximum parenthesis nesting depth accepted in
        ``standard`` mode (``None`` for unlimited). Exceeding it
        raises ``QueryParseError`` pointing at the offending ``(``.

        In ``standard`` mode the configuration is validated eagerly:
        aliases colliding with reserved logical words, aliases
        targeting unknown tokens, or aliases redefining canonical
        operators raise ``QueryBuilderConfigError`` here instead of
        failing (or being silently ignored) at parse time.
        """
        if mode not in ("legacy", "standard"):
            raise QueryBuilderConfigError(
                f"Unknown mode {mode!r}, expected 'legacy' or 'standard'"
            )
        self.table = table
        self.mode = mode
        self.max_depth = max_depth
        # Either override all fields or none.
        if field_aliases:
            self.fields = {k: table[v] for k, v in field_aliases.items()}
        else:
            self.fields = {f.name.lower(): f for f in table if f.readable}
        if token_aliases is None:
            token_aliases = QueryBuilder.default_token_aliases
        self.token_aliases = token_aliases
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
        if mode == "standard":
            self._prepare_standard_mode()

    def _prepare_standard_mode(self) -> None:
        """Precompute lookup tables and validate the configuration."""
        # Reserved logical words (canonical tokens plus their aliases),
        # matched case-insensitively in standard mode.
        self._logical_words = {k.lower(): v for k, v in self.tokens_and_or.items()}
        self._not_words = {k.lower() for k in self.tokens_not}
        self._reserved_logical = set(self._logical_words) | self._not_words
        # Operator phrases as tuples of lowered parts, longest first so
        # ``is not null`` wins over ``is``.
        self._op_phrases = sorted(
            (
                (tuple(key.lower().split()), canonical)
                for key, canonical in self.tokens_ops.items()
            ),
            key=lambda item: (-len(item[0]), -len(" ".join(item[0]))),
        )
        # Symbolic operators (``==``, ``<=``, ...) the lexer must
        # recognize even without surrounding whitespace, longest first.
        self._symbol_tokens = sorted(
            (
                key
                for key in self.tokens_ops
                if " " not in key and not re.match(r"\w", key)
            ),
            key=len,
            reverse=True,
        )
        self._validate_standard_config()

    def _validate_standard_config(self) -> None:
        """
        Reject configurations where aliases collide with reserved
        words or with canonical tokens of another class.
        """
        for name in self.fields:
            if name.lower() in self._reserved_logical:
                raise QueryBuilderConfigError(
                    f"Field alias {name!r} conflicts with a reserved "
                    "logical operator; rename the alias"
                )
        canonical_logical = QueryBuilder.tokens_not | QueryBuilder.tokens_and_or
        canonical_all = canonical_logical | QueryBuilder.tokens_ops
        for alias, target in (self.token_aliases or {}).items():
            normalized = " ".join(alias.split()).lower()
            if target not in canonical_all:
                raise QueryBuilderConfigError(
                    f"Token alias {alias!r} targets unknown token {target!r}; "
                    f"expected one of {sorted(canonical_all)}"
                )
            if normalized in QueryBuilder.tokens_and_or and (
                target not in QueryBuilder.tokens_and_or
            ):
                raise QueryBuilderConfigError(
                    f"Token alias {alias!r} conflicts with reserved logical "
                    f"operator {normalized!r} (cannot map it to {target!r})"
                )
            if normalized in QueryBuilder.tokens_not and (
                target not in QueryBuilder.tokens_not
            ):
                raise QueryBuilderConfigError(
                    f"Token alias {alias!r} conflicts with reserved logical "
                    f"operator {normalized!r} (cannot map it to {target!r})"
                )
            if normalized in QueryBuilder.tokens_ops and target != normalized:
                raise QueryBuilderConfigError(
                    f"Token alias {alias!r} redefines canonical operator "
                    f"{normalized!r} as {target!r}"
                )
            if normalized in self._reserved_logical and target not in canonical_logical:
                raise QueryBuilderConfigError(
                    f"Token alias {alias!r} collides with a reserved logical "
                    "operator alias"
                )

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

    def parse(self, text: str):
        """
        Parse ``text`` into a pydal ``Query``.

        Dispatches to the grammar selected by ``self.mode``; returns
        ``self.table.id > 0`` for empty input (i.e. "all rows").
        """
        if self.mode == "standard":
            return self._parse_standard(text)
        return self._parse_legacy(text)

    def _parse_standard(self, text: str):
        """
        Parse ``text`` with the explicit boolean grammar::

            query     := or_expr
            or_expr   := and_expr (or and_expr)*
            and_expr  := not_expr (and not_expr)*
            not_expr  := not not_expr | primary
            primary   := '(' or_expr ')' | predicate
            predicate := field [lower|upper] op [value (',' value)*]

        ``not`` binds tighter than ``and``, which binds tighter than
        ``or``; parentheses override precedence. The input is lexed
        in one pass, so every ``QueryParseError`` carries character
        offsets into the original string.
        """
        if self.debug:
            print("PARSING", repr(text))
        parser = _StandardParser(self, text)
        if self.debug:
            print("TOKENS", parser.tokens)
        query = parser.parse()
        return query if query is not None else self.table.id > 0

    def _lex(self, text: str) -> "List[_Token]":
        """
        Split ``text`` into tokens, preserving original offsets.

        Double-quoted strings (with ``\\"`` escapes) become ``string``
        tokens whose content is never interpreted — ``and``, ``or``,
        commas and parentheses inside them are data, not syntax.
        Symbolic operators (``==``, ``<=``, ...) are recognized even
        without surrounding whitespace.
        """
        tokens: List[_Token] = []
        symbols = self._symbol_tokens
        i, n = 0, len(text)
        while i < n:
            c = text[i]
            if c.isspace():
                i += 1
                continue
            if c == "(":
                tokens.append(_Token("lparen", c, i, i + 1))
                i += 1
                continue
            if c == ")":
                tokens.append(_Token("rparen", c, i, i + 1))
                i += 1
                continue
            if c == ",":
                tokens.append(_Token("comma", c, i, i + 1))
                i += 1
                continue
            if c == '"':
                j = i + 1
                prev_c: Optional[str] = None
                closed = False
                while j < n:
                    if text[j] == '"' and prev_c != "\\":
                        closed = True
                        break
                    prev_c = text[j]
                    j += 1
                if not closed:
                    raise QueryParseError("Unterminated string literal", i, n)
                tokens.append(_Token("string", text[i + 1 : j], i, j + 1))
                i = j + 1
                continue
            symbol = next((s for s in symbols if text.startswith(s, i)), None)
            if symbol:
                tokens.append(_Token("op", symbol, i, i + len(symbol)))
                i += len(symbol)
                continue
            j = i
            while j < n:
                if text[j].isspace() or text[j] in '(),"':
                    break
                if any(text.startswith(s, j) for s in symbols):
                    break
                j += 1
            tokens.append(_Token("word", text[i:j], i, j))
            i = j
        tokens.append(_Token("eof", "", n, n))
        return tokens

    def _parse_legacy(self, text: str):
        """
        Parse ``text`` with the historical left-to-right grammar::

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


class _Token(NamedTuple):
    """One lexed token; ``start``/``end`` index the original input."""

    kind: str  # "word", "string", "op", "lparen", "rparen", "comma", "eof"
    text: str
    start: int
    end: int

    def describe(self) -> str:
        return "end of input" if self.kind == "eof" else repr(self.text)


class _StandardParser:
    """
    Recursive-descent parser for ``QueryBuilder``'s standard mode.

    Consumes the token stream produced by ``QueryBuilder._lex``;
    precedence is ``not`` > ``and`` > ``or`` with parentheses as an
    escape hatch. All errors reference offsets in the original input.
    """

    def __init__(self, builder: "QueryBuilder", text: str):
        self.builder = builder
        self.text = text
        self.tokens = builder._lex(text)
        self.pos = 0

    def peek(self) -> _Token:
        return self.tokens[self.pos]

    def error(self, message: str, token: _Token) -> None:
        raise QueryParseError(message, token.start, token.end)

    def parse(self):
        if self.peek().kind == "eof":
            return None
        query = self.parse_or(depth=0)
        token = self.peek()
        if token.kind != "eof":
            self.error(f"Unexpected {token.describe()}", token)
        return query

    def parse_or(self, depth: int):
        query = self.parse_and(depth)
        while self._is_logical("or"):
            self.pos += 1
            query = query | self.parse_and(depth)
        return query

    def parse_and(self, depth: int):
        query = self.parse_not(depth)
        while self._is_logical("and"):
            self.pos += 1
            query = query & self.parse_not(depth)
        return query

    def parse_not(self, depth: int):
        token = self.peek()
        if token.kind == "word" and token.text.lower() in self.builder._not_words:
            self.pos += 1
            # Recursion (not a loop) keeps consecutive ``not``s working.
            return ~self.parse_not(depth)
        return self.parse_primary(depth)

    def parse_primary(self, depth: int):
        token = self.peek()
        if token.kind == "lparen":
            max_depth = self.builder.max_depth
            if max_depth is not None and depth + 1 > max_depth:
                self.error(
                    f"Parenthesis nesting exceeds the configured "
                    f"maximum depth of {max_depth}",
                    token,
                )
            self.pos += 1
            query = self.parse_or(depth + 1)
            closing = self.peek()
            if closing.kind != "rparen":
                self.error(f"Expected ')', found {closing.describe()}", closing)
            self.pos += 1
            return query
        return self.parse_predicate()

    def parse_predicate(self):
        builder = self.builder
        token = self.peek()
        if token.kind != "word":
            self.error(f"Expected a field name, found {token.describe()}", token)
        name = token.text.lower()
        if name in builder._reserved_logical:
            self.error(
                f"Reserved word {token.text!r} cannot be used as a field name",
                token,
            )
        field = builder.fields.get(name, builder.fields.get(token.text))
        if field is None:
            self.error(f"Unknown field {token.text!r}", token)
        self.pos += 1
        is_text = field.type in ("string", "text", "blob")
        has_contains = is_text or field.type.startswith("list:")
        op, op_start, op_end = self._match_operator()
        # ``lower``/``upper`` modify the field, then expect another op.
        if is_text and op == "lower":
            field = field.lower()
            op, op_start, op_end = self._match_operator()
        elif is_text and op == "upper":
            field = field.upper()
            op, op_start, op_end = self._match_operator()
        if op == "is null":
            return field == None  # noqa: E711
        if op == "is not null":
            return field != None  # noqa: E711
        if field.type == "boolean" and op == "is true":
            return field == True  # noqa: E712
        if field.type == "boolean" and op == "is false":
            return field == False  # noqa: E712
        value = self._match_value()
        validate(field, value)
        if op == "==":
            return field == value
        if op == "!=":
            return field != value
        if op == "<":
            return field < value
        if op == ">":
            return field > value
        if op == "<=":
            return field <= value
        if op == ">=":
            return field >= value
        if has_contains and op == "contains":
            return field.contains(value)
        if is_text and op == "startswith":
            return field.startswith(value)
        if op == "belongs":
            values = [value]
            while self.peek().kind == "comma":
                self.pos += 1
                item = self._match_value()
                validate(field, item)
                values.append(item)
            return field.belongs(values)
        raise QueryParseError(
            f"Operator {op!r} is not supported for field {field.name!r}",
            op_start,
            op_end,
        )

    def _is_logical(self, canonical: str) -> bool:
        token = self.peek()
        return (
            token.kind == "word"
            and self.builder._logical_words.get(token.text.lower()) == canonical
        )

    def _match_operator(self) -> Tuple[str, int, int]:
        """Longest-match an operator phrase; return canonical + span."""
        for parts, canonical in self.builder._op_phrases:
            if self._match_parts(parts):
                start = self.tokens[self.pos].start
                self.pos += len(parts)
                return canonical, start, self.tokens[self.pos - 1].end
        token = self.peek()
        self.error(f"Expected an operator, found {token.describe()}", token)

    def _match_parts(self, parts: Tuple[str, ...]) -> bool:
        for offset, part in enumerate(parts):
            index = self.pos + offset
            if index >= len(self.tokens):
                return False
            token = self.tokens[index]
            if token.kind == "word" and token.text.lower() == part:
                continue
            if token.kind == "op" and token.text == part:
                continue
            return False
        return True

    def _match_value(self) -> str:
        token = self.peek()
        if token.kind == "string":
            self.pos += 1
            return token.text
        if token.kind == "word":
            if token.text.lower() in self.builder._reserved_logical:
                self.error(
                    f"Reserved word {token.text!r} must be quoted "
                    "to be used as a value",
                    token,
                )
            self.pos += 1
            return token.text
        self.error(f"Expected a value, found {token.describe()}", token)
