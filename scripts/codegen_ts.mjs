
#!/usr/bin/env node
/* JSON Schema -> TypeScript interfaces generator.
 * Usage:
 *   node scripts/codegen_ts.mjs --schemas schemas --out-ts src/models_ts
 */
import { promises as fs } from 'fs';
import path from 'path';

const PRIMS = { string: 'string', number: 'number', integer: 'number', boolean: 'boolean' };

function toCamel(name) {
  return (name || 'Model').replace(/[^0-9a-zA-Z]+/g, ' ').trim()
    .split(/\s+/).map(s => s.charAt(0).toUpperCase() + s.slice(1)).join('').replace(/^\d/, s => 'N'+s);
}

function tsOf(schema, nameHint, models) {
  if (schema && schema.enum) {
    return schema.enum.map(v => JSON.stringify(v)).join(' | ');
  }
  let t = schema?.type;
  if (Array.isArray(t)) {
    const nn = t.filter(x => x!=='null');
    if (nn.length === 1) return tsOf({ ...schema, type: nn[0] }, nameHint, models) + ' | null';
    return 'any';
  }
  if (PRIMS[t]) return PRIMS[t];
  if (t === 'array') return (schema.items ? `Array<${tsOf(schema.items, nameHint+'Item', models)}>` : 'any[]');
  if (t === 'object') {
    const title = (()=>{
      let cand = toCamel(nameHint);
      let i=2; while(models[cand]) { cand = toCamel(nameHint)+i; i++; }
      return cand;
    })();
    const props = schema.properties || {};
    const req = new Set(schema.required || []);
    models[title] = { props: Object.entries(props).map(([k,v])=>({key:k, type: tsOf(v, `${title}_${k}`, models), opt: !req.has(k)})) };
    return title;
  }
  return 'any';
}

async function main() {
  const args = process.argv.slice(2);
  const get = (flag, d) => {
    const i = args.indexOf(flag);
    return i>=0 ? args[i+1] : d;
  };
  const schemas = get('--schemas','schemas');
  const out = get('--out-ts','src/models_ts');
  await fs.mkdir(out, { recursive: true });

  const files = (await fs.readdir(schemas)).filter(f => f.endsWith('.schema.json'));
  for (const fn of files) {
    const sch = JSON.parse(await fs.readFile(path.join(schemas, fn), 'utf-8'));
    const title = sch.title || fn.replace('.schema.json','');
    const mainName = toCamel(title);
    const models = {};
    if (sch.type === 'object') {
      const props = sch.properties || {};
      const req = new Set(sch.required || []);
      const fields = Object.entries(props).map(([k,v])=>{
        let t = tsOf(v, `${mainName}_${k}`, models);
        const opt = !req.has(k);
        return {key:k, type:t, opt};
      });
      models[mainName] = { props: fields };
    } else {
      models[mainName] = { props: [{key:'value', type:'any', opt:false}] };
    }
    let code = `// generated from ${fn}\n`;
    const order = Object.keys(models);
    for (const name of order) {
      code += `export interface ${name} {\n`;
      for (const p of models[name].props) {
        code += `  ${p.key}${p.opt?'?':''}: ${p.type};\n`;
      }
      code += `}\n\n`;
    }
    const outPath = path.join(out, fn.replace('.schema.json','.ts'));
    await fs.writeFile(outPath, code, 'utf-8');
    console.log('generated:', outPath);
  }
}
main().catch(e => { console.error(e); process.exit(1); });
