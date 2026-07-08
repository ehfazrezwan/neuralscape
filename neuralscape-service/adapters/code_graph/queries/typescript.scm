; Tree-sitter query for TypeScript/JavaScript code structure extraction (E3)
; Extracts symbols (functions, classes, methods, imports) and edges (CALLS, IMPORTS, DEFINES, INHERITS).

; ── Definitions ──────────────────────────────────────────────────

(function_declaration
  name: (identifier) @function.def
) @function.scope

(method_definition
  name: (property_identifier) @method.def
) @method.scope

(class_declaration
  name: (type_identifier) @class.def
  heritage: (class_heritage)? @class.heritage
) @class.scope

(interface_declaration
  name: (type_identifier) @interface.def
) @interface.scope

(type_alias_declaration
  name: (type_identifier) @type.def
)

; Arrow functions assigned to variables
(lexical_declaration
  (variable_declarator
    name: (identifier) @function.def
    value: (arrow_function)
  )
)

; ── Imports ──────────────────────────────────────────────────────

(import_statement
  source: (string) @import.source
)

(import_clause
  (identifier) @import.default
)

(named_imports
  (import_specifier
    name: (identifier) @import.named
  )
)

; ── Exports ──────────────────────────────────────────────────────

(export_statement
  declaration: [
    (function_declaration name: (identifier) @export.function)
    (class_declaration name: (type_identifier) @export.class)
  ]
)

; ── Calls ────────────────────────────────────────────────────────

(call_expression
  function: [
    (identifier) @call.function
    (member_expression property: (property_identifier) @call.method)
  ]
)

; ── Extends/Implements ───────────────────────────────────────────

(class_heritage
  (extends_clause
    value: (identifier) @extends.class
  )
)

(class_heritage
  (implements_clause
    (type_identifier) @implements.interface
  )
)
