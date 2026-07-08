; Tree-sitter query for Java code structure extraction (E3)
; Extracts symbols (classes, interfaces, methods, fields) and edges.

; ── Definitions ──────────────────────────────────────────────────

(class_declaration
  name: (identifier) @class.def
  superclass: (superclass (type_identifier) @extends.class)?
  interfaces: (super_interfaces (type_list (type_identifier) @implements.interface))?
) @class.scope

(interface_declaration
  name: (identifier) @interface.def
) @interface.scope

(method_declaration
  name: (identifier) @method.def
) @method.scope

(constructor_declaration
  name: (identifier) @constructor.def
) @constructor.scope

(field_declaration
  declarator: (variable_declarator
    name: (identifier) @field.def
  )
)

; ── Imports ──────────────────────────────────────────────────────

(import_declaration
  (scoped_identifier) @import.class
)

(import_declaration
  (scoped_identifier
    (identifier) @import.name
  )
)

; ── Calls ────────────────────────────────────────────────────────

(method_invocation
  name: (identifier) @call.method
)

(object_creation_expression
  type: (type_identifier) @call.constructor
)

; ── Annotations ──────────────────────────────────────────────────

(annotation
  name: (identifier) @annotation.name
)

(marker_annotation
  name: (identifier) @annotation.marker
)
