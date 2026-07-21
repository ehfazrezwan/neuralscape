; Tree-sitter query for Python code structure extraction (E2)
; Extracts symbols (functions, classes, methods, imports) and edges (CALLS, IMPORTS, DEFINES, INHERITS).

; ── Definitions ──────────────────────────────────────────────────

(function_definition
  name: (identifier) @function.def
) @function.scope

(class_definition
  name: (identifier) @class.def
  superclasses: (argument_list)? @class.bases
) @class.scope

(decorated_definition
  (decorator)* @decorator
  definition: [
    (function_definition name: (identifier) @decorated_function.def)
    (class_definition name: (identifier) @decorated_class.def)
  ]
)

; ── Imports ──────────────────────────────────────────────────────

(import_statement
  name: (dotted_name) @import.module
)

(import_from_statement
  module_name: (dotted_name)? @import.from_module
  name: (dotted_name) @import.from_name
)

(aliased_import
  name: (dotted_name) @import.aliased_name
  alias: (identifier) @import.alias
)

; ── Calls ────────────────────────────────────────────────────────

(call
  function: [
    (identifier) @call.function
    (attribute) @call.method
  ]
)

; ── Assignments (for FQN context) ───────────────────────────────

(assignment
  left: (identifier) @assign.target
  right: (_) @assign.value
)

; ── Method definitions (inside classes) ─────────────────────────

(class_definition
  body: (block
    (function_definition
      name: (identifier) @method.def
    ) @method.scope
  )
)
