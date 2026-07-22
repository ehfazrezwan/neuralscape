; Tree-sitter query for Go code structure extraction (E3)
; Extracts symbols (functions, methods, types, interfaces) and edges.

; ── Definitions ──────────────────────────────────────────────────

(function_declaration
  name: (identifier) @function.def
) @function.scope

(method_declaration
  receiver: (parameter_list
    (parameter_declaration
      type: (type_identifier) @method.receiver
    )
  )
  name: (field_identifier) @method.def
) @method.scope

(type_declaration
  (type_spec
    name: (type_identifier) @type.def
  )
)

(interface_type) @interface.scope

; ── Imports ──────────────────────────────────────────────────────

(import_declaration
  (import_spec
    path: (interpreted_string_literal) @import.path
  )
)

(import_declaration
  (import_spec
    name: (package_identifier) @import.alias
    path: (interpreted_string_literal) @import.path
  )
)

; ── Calls ────────────────────────────────────────────────────────

(call_expression
  function: [
    (identifier) @call.function
    (selector_expression
      field: (field_identifier) @call.method
    )
  ]
)

; ── Struct embedding (composition) ───────────────────────────────

(field_declaration_list
  (field_declaration
    type: (type_identifier) @embeds.type
  )
)
