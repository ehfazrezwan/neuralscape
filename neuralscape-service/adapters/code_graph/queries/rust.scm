; Tree-sitter query for Rust code structure extraction (E3)
; Extracts symbols (functions, structs, traits, impls) and edges.

; ── Definitions ──────────────────────────────────────────────────

(function_item
  name: (identifier) @function.def
) @function.scope

(struct_item
  name: (type_identifier) @struct.def
) @struct.scope

(enum_item
  name: (type_identifier) @enum.def
) @enum.scope

(trait_item
  name: (type_identifier) @trait.def
) @trait.scope

(impl_item
  trait: (type_identifier)? @impl.trait
  type: (type_identifier) @impl.type
) @impl.scope

; Methods inside impl blocks
(impl_item
  body: (declaration_list
    (function_item
      name: (identifier) @method.def
    ) @method.scope
  )
)

; ── Use statements (imports) ─────────────────────────────────────

(use_declaration
  argument: (scoped_identifier) @import.path
)

(use_declaration
  argument: (use_wildcard
    (scoped_identifier) @import.wildcard
  )
)

; ── Calls ────────────────────────────────────────────────────────

(call_expression
  function: [
    (identifier) @call.function
    (field_expression
      field: (field_identifier) @call.method
    )
    (scoped_identifier) @call.scoped
  ]
)

; ── Macro invocations ────────────────────────────────────────────

(macro_invocation
  macro: (identifier) @macro.call
)
