/**
 * Category taxonomy for NeuralScape memory classification.
 *
 * This is the client-side source of truth for the 13 NeuralScape categories,
 * organized into 5 types. It mirrors schemas.py on the server side.
 *
 * Edit this file to add, rename, or reorganize categories. Keep in sync
 * with MEMORY_CATEGORIES and CATEGORY_VAULT_PATHS in neuralscape-service/schemas.py.
 */

export interface Category {
  id: string;
  label: string;
  description: string;
  scope: "global" | "project" | "flexible";
  vaultFolder: string;
}

export interface CategoryType {
  id: string;
  label: string;
  categories: Category[];
}

export const CATEGORY_TYPES: CategoryType[] = [
  {
    id: "semantic",
    label: "Semantic",
    categories: [
      {
        id: "preference",
        label: "Preferences",
        description: "User preferences: language, editor, code style, communication style",
        scope: "global",
        vaultFolder: "Semantic/Preferences",
      },
      {
        id: "personal_fact",
        label: "Personal Facts",
        description: "Personal details: name, timezone, role, team",
        scope: "global",
        vaultFolder: "Semantic/Personal-Facts",
      },
      {
        id: "technical_skill",
        label: "Technical Skills",
        description: "Known technologies, proficiency levels",
        scope: "global",
        vaultFolder: "Semantic/Technical-Skills",
      },
      {
        id: "domain_knowledge",
        label: "Domain Knowledge",
        description: "Industry/domain-specific knowledge",
        scope: "global",
        vaultFolder: "Semantic/Domain-Knowledge",
      },
    ],
  },
  {
    id: "project",
    label: "Project",
    categories: [
      {
        id: "tech_stack",
        label: "Tech Stack",
        description: "Project technology choices",
        scope: "project",
        vaultFolder: "Project/Tech-Stack",
      },
      {
        id: "convention",
        label: "Conventions",
        description: "Coding conventions, naming, file structure",
        scope: "project",
        vaultFolder: "Project/Conventions",
      },
      {
        id: "architecture",
        label: "Architecture",
        description: "Design decisions, module boundaries, API patterns",
        scope: "project",
        vaultFolder: "Project/Architecture",
      },
      {
        id: "dependency",
        label: "Dependencies",
        description: "Packages, versions, compatibility notes",
        scope: "project",
        vaultFolder: "Project/Dependencies",
      },
    ],
  },
  {
    id: "episodic",
    label: "Episodic",
    categories: [
      {
        id: "decision",
        label: "Decisions",
        description: "Decisions made with rationale",
        scope: "flexible",
        vaultFolder: "Episodic/Decisions",
      },
      {
        id: "interaction",
        label: "Interactions",
        description: "Notable past interactions/events",
        scope: "flexible",
        vaultFolder: "Episodic/Interactions",
      },
    ],
  },
  {
    id: "procedural",
    label: "Procedural",
    categories: [
      {
        id: "workflow",
        label: "Workflows",
        description: "Git flow, CI/CD, deployment, review process",
        scope: "flexible",
        vaultFolder: "Procedural/Workflows",
      },
      {
        id: "procedure",
        label: "Procedures",
        description: "Step-by-step how-to patterns",
        scope: "flexible",
        vaultFolder: "Procedural/Procedures",
      },
    ],
  },
  {
    id: "working",
    label: "Working",
    categories: [
      {
        id: "task_context",
        label: "Recent Context",
        description: "Current task, recent changes, blockers",
        scope: "flexible",
        vaultFolder: "Working/Task-Context",
      },
    ],
  },
];

/** Flat array of all categories in display order. */
export const CATEGORIES: Category[] = CATEGORY_TYPES.flatMap((t) => t.categories);

/** Category display order (IDs only). */
export const CATEGORY_ORDER: string[] = CATEGORIES.map((c) => c.id);

/** Map of category ID → display label. */
export const CATEGORY_LABELS: Record<string, string> = Object.fromEntries(
  CATEGORIES.map((c) => [c.id, c.label])
);

export function getCategoryById(id: string): Category | undefined {
  return CATEGORIES.find((c) => c.id === id);
}

export function getVaultFolder(categoryId: string): string {
  return getCategoryById(categoryId)?.vaultFolder ?? `Uncategorized/${categoryId}`;
}

export function getCategoryType(categoryId: string): CategoryType | undefined {
  return CATEGORY_TYPES.find((t) => t.categories.some((c) => c.id === categoryId));
}
