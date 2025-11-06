# Curriculum Graph Schema

The GraphQL schema in this directory (`ai-infra.graphql`) models relationships between roles, modules, projects, assessments, and solutions.

## Use Cases

- Visualize cross-role progression and dependencies.
- Power dashboards or portals that show learner pathways.
- Export module/project data to LMS integrations.

## Working With the Schema

1. Load the schema into your favorite GraphQL tool (e.g., Apollo Studio, GraphQL Playground).
2. Populate data by generating a JSON file from module/project metadata (`./tools/curriculum.py export-graph modules/ projects/ lessons/ --output graphs/generated.json`).
3. Example query:

```graphql
query ModulesForRole {
  modules(roleSlug: "platform-engineer") {
    id
    title
    projects {
      id
      title
    }
  }
}
```

### Sample Data

`graphs/samples/ai-infra-sample.json` demonstrates how a minimal dataset can represent modules and projects. Use it as a reference when building exporters.

## Next Steps

- Extend the schema with additional fields (e.g., validation status, metadata URL).
- Generate resolvers that pull data from metadata YAML files.
- Use the graph to enforce duplication checks across roles.
