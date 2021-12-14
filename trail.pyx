# cython: language_level=3

cimport cython
import heapq
import re
import typing as t
import uuid
from cpython cimport bool
from datetime import datetime, date
from urllib.parse import quote, unquote


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def alpha(value: str) -> str:
    if not value.isalpha():
        raise ValueError(
            f"Value {value} contains non-alphabetic characters")
    return value


TYPES = {
    "any": (str, r"[^/]+"),
    "str": (str, r"[^/]+"),
    "alpha": (alpha, r"[A-Za-z]+"),
    "path": (str, r"[^/]?.*?"),
    "number": (float, r"-?(?:\d+(?:\.\d*)?|\.\d+)"),
    "float": (float, r"-?(?:\d+(?:\.\d*)?|\.\d+)"),
    "int": (int, r"-?\d+"),
    "ymd": (
        parse_date,
        r"([12]\d{3}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01]))",
    ),
    "uuid": (
        uuid.UUID,
        r"[A-Fa-f0-9]{8}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-"
        r"[A-Fa-f0-9]{4}-[A-Fa-f0-9]{12}"
    ),
}



SLUG = re.compile(r'^([a-zA-Z]+)(:(.+))?$')


cdef class SlugPart:
    pass


cdef class StrSlug(SlugPart):
    cdef public str value
    cdef public str pattern

    def __cinit__(self, value):
        self.value = value
        self.pattern = re.escape(value)

    def __repr__(self):
        return f'<StringPart: {self.value!r}>'


cdef class DynamicSlug(SlugPart):
    cdef str value
    cdef public str name
    cdef public str pattern
    cdef public converter

    def __cinit__(self, value):
        self.value = value
        result = SLUG.match(value)
        self.name = result.group(1)
        cdef str _type = result.group(3) or 'str'

        ptype = TYPES.get(_type)
        if ptype is None:
            pattern = _type
            self.converter = str
        else:
            self.converter, pattern = ptype

        self.pattern = f"(?P<{self.name}>{pattern})"

    def __repr__(self):
        return f'<DynamicPart: {self.value!r} -> {self.pattern}>'


cdef class Slug:
    cdef readonly unsigned int complexity
    cdef readonly object compiled
    cdef readonly bool exact
    cdef public bool final
    cdef public dict namespace
    cdef public dict converters

    def __cinit__(self, parts, final: bool):
        self.complexity = len(parts)
        self.namespace = None
        self.final = final
        self.exact = self.complexity == 1 and isinstance(parts[0], StrSlug)
        if self.exact:
            self.compiled = parts[0].value
        else:
            self.converters = {}
            self.compiled = re.compile(
                "^{}$".format(
                    ''.join((part.pattern for part in parts))
                )
            )
            self.converters = {
                part.name: part.converter
                for part in parts if (
                        isinstance(part, DynamicSlug)
                        and part.converter != str
                )
            }

    cpdef dict convert(self, dict variables):
        for name, caster in self.converters.items():
            variables[name] = caster(variables[name])
        return variables

    def __eq__(self, slug):
        return self.compiled == slug.compiled and self.final == slug.final

    def __repr__(self):
        return f'<Slug: {self.compiled!r} final:{self.final}>'


cdef list _path_to_parts(str path, str delimiter="/"):
    path = path.strip('/')
    if not path:
        return None

    cdef list parts
    cdef list slugs = []
    cdef unsigned int start, current = 0
    cdef unsigned int end = len(path)

    parts = []
    while start < end:
        if path[current] == '/':
            slugs.append(Slug(parts, False))
            parts = []
            current += 1
        elif path[current] == '{':
            start += 1
            current += 1
            while current <= end:
                current += 1
                if path[current] == '}':
                    parts.append(DynamicSlug(path[start:current]))
                    current += 1
                    break
        else:
            while current < end and path[current] not in ('{', '/'):
                current += 1
            else:
                parts.append(StrSlug(path[start:current]))

        start = current
    else:
        slugs.append(Slug(parts, True))

    return slugs


cdef class Node:

    cdef readonly Slug slug
    cdef dict exact
    cdef list dynamic

    def __cinit__(self, Slug slug):
        self.slug = slug
        self.exact = {}
        self.dynamic = []

    cpdef add(self, Slug part):
        if part.exact:
            found = self.exact.get(part.compiled)
            if found is None:
                found = self.exact[part.compiled] = Node(part)
            elif part.final and not found.slug.final:
                found.slug.final = True
            return found

        for node in self.dynamic:
            if node.slug == part:
                if part.final and not node.slug.final:
                    node.slug.final = True
                return node

        found = Node(part)
        heapq.heappush(self.dynamic, found)
        return found

    cpdef tuple match(self, list parts, dict variables):
        cdef bool last = (len(parts) == 1)

        if self.exact:
            node = self.exact.get(parts[0])
            if node is not None:
                if last:
                    if node.slug.final:
                        return variables, node.slug.namespace
                    else:
                        return
                return node.match(parts[1:], variables)

        if self.dynamic:
            for node in self.dynamic:
                matched = node.slug.compiled.match(parts[0])
                if matched:
                    if last:
                        if node.slug.final:
                            variables.update(
                                node.slug.convert(matched.groupdict())
                            )
                            return variables, node.slug.namespace
                        else:
                            return
                    found = node.match(parts[1:], variables)
                    if found is not None:
                        variables.update(
                            node.slug.convert(matched.groupdict())
                        )
                        return found

    def __lt__(self, node):
        return self.slug.complexity < node.slug.complexity

    def __repr__(self):
        return repr(self.slug)


cdef class Routes:

    cdef Node root

    def __cinit__(self):
        self.root = Node(None)

    cdef void _add(self, str path, dict namespace):
        slugs = _path_to_parts(path)
        if not slugs:
            return
        node = self.root
        for slug in slugs:
            node = node.add(slug)
        if node.slug.namespace is not None:
            node.slug.namespace.update(namespace)
        else:
            node.slug.namespace = namespace

    cpdef tuple match(self, str path):
        path = path.strip('/')
        cdef list parts = path.split('/')
        return self.root.match(parts, {})

    def add(self, path: str, **namespace):
        self._add(path, namespace)
