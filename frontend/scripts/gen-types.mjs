// Generate src/api/types.gen.ts from the ScreeningResult JSON Schema (single source of truth:
// pipeline/mukha_pipeline/schema/models.py -> screening_result.v1.schema.json -> this file).
//
//   npm run gen:types
//
// Property-level `title`s are stripped before compiling so json-schema-to-typescript does not
// emit one alias per field (e.g. `type WlImage = string | null`). Named types come from $defs.
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { compile } from 'json-schema-to-typescript';

const here = dirname(fileURLToPath(import.meta.url));
const schemaPath = resolve(here, '../../pipeline/mukha_pipeline/schema/screening_result.v1.schema.json');
const outPath = resolve(here, '../src/api/types.gen.ts');

const schema = JSON.parse(await readFile(schemaPath, 'utf8'));

function stripPropertyTitles(node) {
  if (Array.isArray(node)) return node.forEach(stripPropertyTitles);
  if (!node || typeof node !== 'object') return;
  if (node.properties && typeof node.properties === 'object') {
    for (const prop of Object.values(node.properties)) {
      if (prop && typeof prop === 'object') delete prop.title;
    }
  }
  for (const value of Object.values(node)) stripPropertyTitles(value);
}

stripPropertyTitles(schema);
schema.title = 'ScreeningResult';
delete schema.$id;

const ts = await compile(schema, 'ScreeningResult', {
  bannerComment: `/* eslint-disable */
/**
 * GENERATED FILE — do not edit by hand.
 * Source: pipeline/mukha_pipeline/schema/screening_result.v1.schema.json (ScreeningResult v${schema['x-schema-version'] ?? '1'})
 * Regenerate with: npm run gen:types
 */`,
  additionalProperties: false,
  unreachableDefinitions: true,
  declareExternallyReferenced: true,
  format: true,
  style: { singleQuote: true, printWidth: 100 },
  cwd: dirname(schemaPath),
});

await writeFile(outPath, ts, 'utf8');
console.log(`wrote ${outPath}`);
